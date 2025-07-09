from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import uvicorn
from typing import Optional, List, Dict, Any
import os
import json
import re
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import math

# Configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
MODEL_NAME = "mistralai/mistral-small-3.2-24b-instruct:free"
FAISS_DIR = "faiss_store"

app = FastAPI(title="YouTube Video Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active chatbots
active_chatbots: Dict[str, ConversationalRetrievalChain] = {}

# Pydantic models
class VideoRequest(BaseModel):
    youtube_url: str

class ChatRequest(BaseModel):
    video_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []

# Custom LLM for OpenRouter
class OpenRouterLLM(LLM):
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"OpenRouter error {response.status_code}: {response.text}")

# Utility functions
def extract_video_id(youtube_url: str) -> Optional[str]:
    match = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", youtube_url)
    return match.group(1) if match else None

def get_video_info(video_id: str) -> Dict[str, str]:
    """Get video title and description using YouTube API or web scraping"""
    try:
        # Simple web scraping approach (you might want to use YouTube API for production)
        import requests
        from bs4 import BeautifulSoup
        
        url = f"https://www.youtube.com/watch?v={video_id}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title = soup.find('title').text.replace(' - YouTube', '') if soup.find('title') else f"Video {video_id}"
        
        return {
            "title": title,
            "description": ""  # You can extract description if needed
        }
    except:
        return {"title": f"Video {video_id}", "description": ""}

def download_transcript(video_id: str) -> List[Dict[str, Any]]:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return [{"timestamp": float(entry["start"]), "text": entry["text"]} for entry in transcript]
    except Exception as e:
        print(f"Transcript download error: {e}")
        return []

def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def create_enhanced_chunks(transcript: List[Dict[str, Any]], video_info: Dict[str, str]) -> List[Document]:
    """Create enhanced chunks with better context and metadata"""
    if not transcript:
        return []
    
    # Use RecursiveCharacterTextSplitter for better chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Increased chunk size for better context
        chunk_overlap=100,  # Overlap to maintain context between chunks
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    
    documents = []
    
    # Group transcript entries into larger segments for better context
    segment_duration = 60  # 60 seconds per segment
    current_segment = []
    current_start_time = 0
    
    for entry in transcript:
        if not current_segment:
            current_start_time = entry["timestamp"]
        
        current_segment.append(entry)
        
        # Check if we should close this segment
        if (entry["timestamp"] - current_start_time >= segment_duration or 
            entry == transcript[-1]):
            
            # Combine text from current segment
            segment_text = " ".join([e["text"] for e in current_segment])
            
            # Clean up the text
            segment_text = re.sub(r'\s+', ' ', segment_text).strip()
            
            if segment_text:
                # Add context about video and timing
                start_time = format_timestamp(current_start_time)
                end_time = format_timestamp(entry["timestamp"])
                
                contextual_text = f"Video: {video_info['title']}\n"
                contextual_text += f"Time: {start_time} - {end_time}\n"
                contextual_text += f"Content: {segment_text}"
                
                # Split the contextual text if it's too long
                chunks = text_splitter.split_text(contextual_text)
                
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "video_id": extract_video_id(video_info.get("url", "")),
                            "start_time": current_start_time,
                            "end_time": entry["timestamp"],
                            "formatted_time": f"{start_time} - {end_time}",
                            "chunk_index": len(documents) + i,
                            "video_title": video_info["title"]
                        }
                    )
                    documents.append(doc)
            
            current_segment = []
    
    return documents

def load_or_create_vectorstore(youtube_url: str) -> tuple[FAISS, List[Document]]:
    video_id = extract_video_id(youtube_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    os.makedirs(FAISS_DIR, exist_ok=True)
    vectorstore_path = os.path.join(FAISS_DIR, f"{video_id}_vectorstore")
    
    # Use a better embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    if os.path.exists(vectorstore_path):
        print(f"Loading existing vectorstore for video {video_id}...")
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        with open(os.path.join(vectorstore_path, "documents.json"), "r", encoding="utf-8") as f:
            doc_data = json.load(f)
            documents = [Document(page_content=d["content"], metadata=d["metadata"]) for d in doc_data]
        return vectorstore, documents

    print(f"Downloading transcript for video {video_id}...")
    transcript = download_transcript(video_id)
    if not transcript:
        raise Exception("Failed to download transcript")

    print("Getting video info...")
    video_info = get_video_info(video_id)
    video_info["url"] = youtube_url

    print("Creating enhanced chunks...")
    documents = create_enhanced_chunks(transcript, video_info)

    print("Creating vectorstore...")
    if documents:
        vectorstore = FAISS.from_documents(documents, embeddings)
        os.makedirs(vectorstore_path, exist_ok=True)
        vectorstore.save_local(vectorstore_path)
        
        # Save documents for later loading
        doc_data = [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]
        with open(os.path.join(vectorstore_path, "documents.json"), "w", encoding="utf-8") as f:
            json.dump(doc_data, f, ensure_ascii=False, indent=2, default=str)
    else:
        raise Exception("No documents created from transcript")

    return vectorstore, documents

def create_chatbot(youtube_url: str) -> ConversationalRetrievalChain:
    vectorstore, documents = load_or_create_vectorstore(youtube_url)

    llm = OpenRouterLLM(
        api_key=OPENROUTER_API_KEY, 
        model=MODEL_NAME,
        temperature=0.3,  # Lower temperature for more focused responses
        max_tokens=1500
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Enhanced prompt template with better instructions
    template = """You are a helpful AI assistant that answers questions based on a YouTube video transcript. You have access to the video content with timestamps and context.

INSTRUCTIONS:
1. Use the provided context to answer questions accurately and comprehensively
2. If you reference specific parts of the video, mention the timestamp when relevant
3. If you don't know the answer based on the transcript, say so clearly
4. Provide detailed answers when possible, using multiple parts of the transcript if relevant
5. Maintain context from the conversation history

CONTEXT FROM VIDEO:
{context}

CONVERSATION HISTORY:
{chat_history}

QUESTION: {question}

ANSWER: Please provide a comprehensive answer based on the video content. If you reference specific information, include the timestamp when relevant."""

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )

    # Create retriever with better search parameters
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 6,  # Retrieve more documents for better context
            "score_threshold": 0.3  # Lower threshold to include more relevant results
        }
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,  # Return source documents for better debugging
        output_key="answer",
        max_tokens_limit=3000  # Increased token limit for better context
    )

    return chain

# API Routes
@app.get("/")
async def root():
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="HTML template not found")

@app.post("/process-video")
async def process_video(request: VideoRequest):
    try:
        video_id = extract_video_id(request.youtube_url)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")

        chatbot = create_chatbot(request.youtube_url)
        active_chatbots[video_id] = chatbot

        return {"video_id": video_id, "success": True}

    except Exception as e:
        return {"video_id": "", "success": False, "error": str(e)}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if request.video_id not in active_chatbots:
            raise HTTPException(status_code=404, detail="Video not processed yet")

        chatbot = active_chatbots[request.video_id]
        result = chatbot.invoke({"question": request.question})

        sources = []
        if result.get("source_documents"):
            for doc in result["source_documents"][:3]:  # Top 3 sources
                source_text = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                if doc.metadata.get("formatted_time"):
                    source_text = f"[{doc.metadata['formatted_time']}] {source_text}"
                sources.append(source_text)

        return ChatResponse(
            answer=result["answer"],
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video/{video_id}/status")
async def get_video_status(video_id: str):
    return {
        "video_id": video_id,
        "processed": video_id in active_chatbots,
        "active_videos": list(active_chatbots.keys())
    }

@app.delete("/video/{video_id}")
async def remove_video(video_id: str):
    if video_id in active_chatbots:
        del active_chatbots[video_id]
        return {"message": f"Video {video_id} removed from active chatbots"}
    else:
        raise HTTPException(status_code=404, detail="Video not found in active chatbots")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)