import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
import gc

from src.graph import run_langgraph_stream
from src.embedder import (
    create_vector_db,
    list_ingested_documents,
    delete_document_collection,
    get_or_create_vectorstore,
    _get_embeddings
)
import src.state as global_state
from rank_bm25 import BM25Okapi
from src.models import (
    QuestionRequest,
    QuestionResponse,
    HealthResponse,
    AnalyticsResponse,
    TopQueriesResponse,
    ChatResetResponse,
    ChatHistoryResponse,
    SystemStatusResponse,
    DocumentListResponse,
)
from src.failure_analytics import summarize_failures, top_problem_queries
from src.state import get_full_chat_history_from_db, clear_chat_history_db

app = FastAPI(
    title="Self-Healing RAG API",
    version="2.0.0",
    description="A self-healing, conversational RAG assistant with adaptive retrieval and reflection."
)

# React frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# HEALTH & SYSTEM STATUS
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return {"status": "ok"}


@app.get("/system/status", response_model=SystemStatusResponse, tags=["System"])
def system_status():
    """Returns the current runtime state of the RAG system instantly."""
    ingested = list_ingested_documents()
    docs_count = sum(d["chunk_count"] for d in ingested)

    llm_name = "llama3.2 (Ollama)"
    if os.environ.get("GROQ_API_KEY"):
        llm_name = f"{os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')} (Groq Cloud)"

    return {
        "status": "ok",
        "total_chunks": docs_count,
        "bm25_ready": global_state.bm25 is not None,
        "vectorstore_ready": global_state.vectorstore is not None,
        "llm_model": llm_name,
        "embedding_model": "FastCloudEmbeddings (Cloud)" if os.environ.get("GROQ_API_KEY") else "mxbai-embed-large (Local)",
        "documents_ingested": len(ingested),
    }


# ─────────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────────

@app.post("/ask", tags=["Chat"])
async def ask(request: QuestionRequest):
    """Stream a response for a given question using the self-healing RAG pipeline."""
    return StreamingResponse(
        run_langgraph_stream(request.question),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",      # Disable nginx proxy buffering on Render
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/chat/history", response_model=ChatHistoryResponse, tags=["Chat"])
def get_chat_history():
    """Returns the full conversation history from persistent storage."""
    history = get_full_chat_history_from_db()
    return {
        "history": history,
        "total_messages": len(history)
    }


@app.post("/chat/reset", response_model=ChatResetResponse, tags=["Chat"])
def reset_chat():
    """Clears the conversation history (both in-memory and SQLite) and entity/mention memory."""
    global_state.chat_history.clear()
    global_state.entity_memory["recent_entities"] = []
    global_state.mention_memory["recent_mentions"] = []
    clear_chat_history_db()
    global_state.save_session()
    return {"status": "success", "message": "Chat history and memory cleared."}


# ─────────────────────────────────────────────
# DOCUMENTS
# ─────────────────────────────────────────────

@app.post("/upload", tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a PDF or text document into the vector database with zero-memory-spike optimization."""
    os.makedirs("Data", exist_ok=True)
    file_path = os.path.join("Data", file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        from src.splitter import split_documents

        embeddings = _get_embeddings()
        chunks = split_documents(file_path, embeddings=embeddings)
        count = len(chunks)

        vectorstore = get_or_create_vectorstore()
        vectorstore.add_documents(chunks)

        new_docs = [c.page_content for c in chunks]
        new_meta = [c.metadata for c in chunks]
        global_state.ALL_DOCS.extend(new_docs)
        global_state.ALL_METADATA.extend(new_meta)

        if global_state.ALL_DOCS:
            tokenized_corpus = [doc.lower().split() for doc in global_state.ALL_DOCS]
            global_state.bm25 = BM25Okapi(tokenized_corpus)

        gc.collect()

        return {
            "status": "success",
            "message": f"Successfully ingested '{file.filename}' into {count} semantic chunks!"
        }
    except Exception as e:
        print(f"[app] Upload error: {e}")
        gc.collect()
        return {"status": "error", "message": str(e)}


@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
def get_documents():
    """Returns a list of all ingested documents with their chunk counts."""
    docs = list_ingested_documents()
    total = sum(d["chunk_count"] for d in docs)
    return {
        "documents": docs,
        "total_chunks": total
    }


@app.delete("/documents/{doc_name}", tags=["Documents"])
def delete_document(doc_name: str):
    """Delete a specific document's embeddings from the vector database."""
    try:
        success = delete_document_collection(doc_name)
        if success:
            if global_state.vectorstore:
                db = global_state.vectorstore.get(include=["documents", "metadatas"])
                global_state.ALL_DOCS = db.get("documents", []) or []
                global_state.ALL_METADATA = db.get("metadatas", []) or []
                if global_state.ALL_DOCS:
                    tokenized_corpus = [doc.lower().split() for doc in global_state.ALL_DOCS]
                    global_state.bm25 = BM25Okapi(tokenized_corpus)
            gc.collect()
            return {"status": "success", "message": f"Deleted '{doc_name}' from knowledge base."}
        else:
            raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────

@app.get("/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
def get_analytics():
    """Returns aggregate failure analytics from persistent failure logs."""
    return summarize_failures()


@app.get("/analytics/top-queries", response_model=TopQueriesResponse, tags=["Analytics"])
def get_top_queries(top_k: int = 10):
    """Returns the most frequently failing queries."""
    queries = top_problem_queries(top_k=top_k)
    return {
        "top_queries": [{"query": q, "count": c} for q, c in queries]
    }