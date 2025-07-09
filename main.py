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
            "messages": [{"role": "user", "content": prompt}]
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

def download_transcript(video_id: str) -> List[Dict[str, Any]]:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return [{"timestamp": float(entry["start"]), "text": entry["text"]} for entry in transcript]
    except Exception as e:
        print(f"Transcript download error: {e}")
        return []

def chunk_transcript(transcript: List[Dict[str, any]], max_chars: int = 500) -> List[str]:
    chunks = []
    current_chunk = ""
    for entry in transcript:
        text = entry["text"].strip().replace("\n", " ")
        if len(current_chunk) + len(text) + 1 <= max_chars:
            current_chunk += " " + text
        else:
            chunks.append(current_chunk.strip())
            current_chunk = text
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def load_or_create_vectorstore(youtube_url: str) -> tuple[FAISS, List[str]]:
    video_id = extract_video_id(youtube_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    os.makedirs(FAISS_DIR, exist_ok=True)
    vectorstore_path = os.path.join(FAISS_DIR, f"{video_id}_vectorstore")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(vectorstore_path):
        print(f"Loading existing vectorstore for video {video_id}...")
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        with open(os.path.join(vectorstore_path, "chunks.json"), "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return vectorstore, chunks

    print(f"Downloading transcript for video {video_id}...")
    transcript = download_transcript(video_id)
    if not transcript:
        raise Exception("Failed to download transcript")

    print("Chunking transcript...")
    chunks = chunk_transcript(transcript)

    print("Creating vectorstore...")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    os.makedirs(vectorstore_path, exist_ok=True)
    vectorstore.save_local(vectorstore_path)
    with open(os.path.join(vectorstore_path, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return vectorstore, chunks

def create_chatbot(youtube_url: str) -> ConversationalRetrievalChain:
    vectorstore, chunks = load_or_create_vectorstore(youtube_url)

    llm = OpenRouterLLM(api_key=OPENROUTER_API_KEY, model=MODEL_NAME)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    template = """You are a helpful assistant answering questions based on a YouTube video transcript.

Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False,
        output_key="answer"
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
            sources = [doc.page_content[:200] + "..." for doc in result["source_documents"][:3]]

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
