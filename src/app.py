from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os

from src.graph import run_langgraph_stream
from src.embedder import create_vector_db
import src.state as global_state
from rank_bm25 import BM25Okapi
from src.models import (
    QuestionRequest,
    QuestionResponse,
    HealthResponse
)

app = FastAPI(
    title="Self-Healing RAG API",
    version="1.0.0"
)

# React frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok"
    }


@app.post("/ask")
async def ask(request: QuestionRequest):
    return StreamingResponse(
        run_langgraph_stream(request.question), 
        media_type="text/event-stream"
    )

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    os.makedirs("Data", exist_ok=True)
    file_path = os.path.join("Data", file.filename)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    try:
        import asyncio
        # Re-ingest (run in thread to prevent blocking event loop)
        vectorstore = await asyncio.to_thread(create_vector_db, file_path)
        count = vectorstore._collection.count()
        
        # Update system state to instantly reflect new docs
        db = vectorstore.get()
        global_state.vectorstore = vectorstore
        global_state.ALL_DOCS = db["documents"]
        global_state.ALL_METADATA = db["metadatas"]
        
        if global_state.ALL_DOCS:
            tokenized_corpus = [doc.lower().split() for doc in global_state.ALL_DOCS]
            global_state.bm25 = BM25Okapi(tokenized_corpus)
        
        return {"status": "success", "message": f"Successfully ingested {file.filename} into {count} semantic chunks!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}