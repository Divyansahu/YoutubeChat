"""
backend.py — FastAPI server for YouTube AI Chat Chrome Extension
================================================================
Wraps your existing Gemini + FAISS + LangChain notebook logic
into two REST endpoints the Chrome Extension calls.

Run locally:
    pip install fastapi uvicorn langchain-google-genai langchain-community
               langchain-classic langchain-text-splitters faiss-cpu supadata
    export GOOGLE_API_KEY="your_key"
    export SUPADATA_API_KEY="your_key"
    uvicorn backend:app --reload --port 8000
"""

from dotenv import load_dotenv
load_dotenv()

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain / Gemini imports (same as your notebook)
from supadata import Supadata, SupadataError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_classic.retrievers import MultiQueryRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yt-ai-backend")

# ── API Keys ──────────────────────────────────────────────────────────────────
# Set these as environment variables or replace directly (not recommended)
GOOGLE_API_KEY  = os.environ.get("GOOGLE_API_KEY",  "YOUR_GOOGLE_API_KEY")
SUPADATA_API_KEY = os.environ.get("SUPADATA_API_KEY", "YOUR_SUPADATA_API_KEY")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ── In-memory store: one FAISS index + chain per video_id ────────────────────
# Structure: { video_id: { "chain": <RunnableSequence>, "title": str } }
video_store: dict = {}

# ── Supadata client ───────────────────────────────────────────────────────────
supadata_client = Supadata(api_key=SUPADATA_API_KEY)

# print("GOOGLE KEY:", os.environ.get("GOOGLE_API_KEY", "NOT FOUND"))
# print("SUPADATA KEY:", os.environ.get("SUPADATA_API_KEY", "NOT FOUND"))
# ── Pydantic request/response models ─────────────────────────────────────────
class ProcessVideoRequest(BaseModel):
    video_id: str

class AskRequest(BaseModel):
    video_id: str
    question: str

class ProcessVideoResponse(BaseModel):
    success: bool
    video_id: str
    message: str
    chunk_count: Optional[int] = None

class AskResponse(BaseModel):
    answer: str
    video_id: str


# ── Core: build the RAG chain for a video (mirrors your notebook exactly) ────
def build_chain_for_video(video_id: str) -> dict:
    """
    1. Fetch transcript via Supadata
    2. Split into chunks (RecursiveCharacterTextSplitter)
    3. Embed with Gemini + store in FAISS
    4. Build MultiQueryRetriever + RAG chain
    Returns dict with keys: chain, chunk_count
    """
    logger.info(f"[{video_id}] Fetching transcript…")

    # Step 1 — Fetch transcript (same as your notebook)
    try:
        transcript = supadata_client.youtube.transcript(
            video_id=video_id,
            text=True
        ).content
    except SupadataError as e:
        raise ValueError(f"Supadata could not fetch transcript: {e}")

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty – video may not have captions.")

    logger.info(f"[{video_id}] Transcript fetched ({len(transcript)} chars). Chunking…")

    # Step 2 — Split (same as your notebook)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.create_documents([transcript])
    logger.info(f"[{video_id}] {len(chunks)} chunks created. Building FAISS…")

    # Step 3 — Embed + FAISS (same as your notebook)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vector_store = FAISS.from_documents(chunks, embeddings)
    logger.info(f"[{video_id}] FAISS index ready.")

    # Step 4 — LLM + MultiQueryRetriever (same as your notebook)
    llm = ChatGoogleGenerativeAI(
        # model="gemini-2.5-flash",
        model="gemini-2.0-flash",
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        llm=llm
    )

    # Step 5 — Prompt (same as your notebook)
    prompt = PromptTemplate(
        template="""
          You are a helpful assistant.
          Answer ONLY from the provided transcript context.
          If the context is insufficient, just say you don't know.

          {context}
          Question: {question}
        """,
        input_variables=["context", "question"]
    )

    # Step 6 — Chain assembly (same as your notebook)
    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel({
        "context": multiquery_retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser

    return {
        "chain": main_chain,
        "chunk_count": len(chunks)
    }


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="YouTube AI Chat Backend",
    description="Gemini + FAISS RAG backend for the YouTube AI Chat Chrome Extension",
    version="1.0.0"
)

# Allow requests from the Chrome extension (content scripts use null origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
async def health():
    return {
        "status": "ok",
        "service": "YouTube AI Chat Backend",
        "indexed_videos": list(video_store.keys())
    }


# ── POST /process-video ───────────────────────────────────────────────────────
@app.post("/process-video", response_model=ProcessVideoResponse)
async def process_video(body: ProcessVideoRequest):
    """
    Called automatically by the Chrome Extension when a new YouTube video is opened.
    Fetches the transcript, chunks it, embeds it, and stores the RAG chain.
    """
    video_id = body.video_id.strip()

    if not video_id:
        raise HTTPException(status_code=400, detail="video_id is required")

    # Already indexed – skip re-processing
    if video_id in video_store:
        logger.info(f"[{video_id}] Already indexed. Skipping.")
        return ProcessVideoResponse(
            success=True,
            video_id=video_id,
            message="Already processed",
            chunk_count=video_store[video_id].get("chunk_count")
        )

    try:
        result = build_chain_for_video(video_id)
        video_store[video_id] = result
        logger.info(f"[{video_id}] ✅ Ready. {result['chunk_count']} chunks indexed.")
        return ProcessVideoResponse(
            success=True,
            video_id=video_id,
            message="Video processed and indexed successfully",
            chunk_count=result["chunk_count"]
        )
    except ValueError as e:
        logger.error(f"[{video_id}] ❌ {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"[{video_id}] ❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# ── POST /ask ─────────────────────────────────────────────────────────────────
@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest):
    """
    Called when the user submits a question in the chat panel.
    Runs the RAG chain for the given video_id.
    """
    video_id = body.video_id.strip()
    question = body.question.strip()

    if not video_id:
        raise HTTPException(status_code=400, detail="video_id is required")
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    # Auto-process if not yet indexed (e.g., user asks before banner is done)
    if video_id not in video_store:
        logger.info(f"[{video_id}] Not indexed yet – indexing on demand…")
        try:
            result = build_chain_for_video(video_id)
            video_store[video_id] = result
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    chain = video_store[video_id]["chain"]

    try:
        logger.info(f"[{video_id}] Invoking chain with: '{question}'")
        answer = chain.invoke(question)
        logger.info(f"[{video_id}] ✅ Answer generated.")
        return AskResponse(answer=answer, video_id=video_id)
    except Exception as e:
        logger.error(f"[{video_id}] Chain error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")


# ── Optional: clear a video from memory ──────────────────────────────────────
@app.delete("/video/{video_id}")
async def clear_video(video_id: str):
    if video_id in video_store:
        del video_store[video_id]
        return {"success": True, "message": f"Cleared {video_id}"}
    return {"success": False, "message": "Video not found in store"}
