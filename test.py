import re
import os
import json
import faiss
import numpy as np
import requests
from typing import List, Dict, Union
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from sentence_transformers import SentenceTransformer

# LangChain imports
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.prompts import PromptTemplate
from typing import Any, List, Optional


# ================================
# CONFIGURATION VARIABLES
# ================================
YOUTUBE_URL = "https://youtu.be/gqvFmK7LpDo?si=4pkfXVZYQrcwUwEw"
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
MODEL_NAME = "mistralai/mistral-small-3.2-24b-instruct:free"
FAISS_DIR = "faiss_store"


# ================================
# CUSTOM LLM FOR OPENROUTER
# ================================
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


# ================================
# UTILS
# ================================
def extract_video_id(youtube_url: str) -> Union[str, None]:
    match = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", youtube_url)
    return match.group(1) if match else None


def download_transcript(video_id: str) -> List[Dict[str, Union[float, str]]]:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return [{"timestamp": float(entry["start"]), "text": entry["text"]} for entry in transcript]
    except TranscriptsDisabled:
        print("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        print("No transcript found for this video.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return []


def chunk_transcript(transcript: List[Dict[str, Union[float, str]]], max_chars: int = 500) -> List[str]:
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


def create_vectorstore(chunks: List[str], video_id: str) -> FAISS:
    """Create FAISS vectorstore from chunks"""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    # Save vectorstore
    os.makedirs(FAISS_DIR, exist_ok=True)
    vectorstore.save_local(os.path.join(FAISS_DIR, f"{video_id}_vectorstore"))
    
    return vectorstore


def load_or_create_vectorstore(youtube_url: str) -> FAISS:
    """Load existing vectorstore or create new one from transcript"""
    video_id = extract_video_id(youtube_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    
    vectorstore_path = os.path.join(FAISS_DIR, f"{video_id}_vectorstore")
    
    # Try to load existing vectorstore
    if os.path.exists(vectorstore_path):
        print(f"Loading existing vectorstore for video {video_id}...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    
    # Create new vectorstore from transcript
    print(f"Downloading transcript for video {video_id}...")
    transcript = download_transcript(video_id)
    if not transcript:
        raise Exception("Failed to download transcript")
    
    print("Chunking transcript...")
    chunks = chunk_transcript(transcript)
    
    print("Creating vectorstore...")
    return create_vectorstore(chunks, video_id)


def create_chatbot(youtube_url: str) -> ConversationalRetrievalChain:
    """Create a chatbot with memory for the given YouTube video"""
    
    # Load or create vectorstore
    vectorstore = load_or_create_vectorstore(youtube_url)
    
    # Create LLM
    llm = OpenRouterLLM(api_key=OPENROUTER_API_KEY, model=MODEL_NAME)
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer" 
    )
    
    # Create prompt template
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
    
    # Create conversational chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False,
        output_key="answer"
    )
    
    return chain


def run_chatbot():
    """Run the interactive chatbot"""
    print("🤖 YouTube Video Chatbot")
    print("=" * 50)
    print(f"Video: {YOUTUBE_URL}")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    try:
        chatbot = create_chatbot(YOUTUBE_URL)
        print("✅ Chatbot ready! Ask questions about the video.\n")
        
        while True:
            question = input("\n❓ You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            try:
                print("\n🤖 Bot: ", end="", flush=True)
                result = chatbot.invoke({"question": question})
                print(result["answer"])
                
                # Show sources if available
                if result.get("source_documents"):
                    print("\n📚 Sources:")
                    for i, doc in enumerate(result["source_documents"][:2], 1):
                        print(f"  {i}. {doc.page_content[:100]}...")
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")


if __name__ == "__main__":
    run_chatbot()
