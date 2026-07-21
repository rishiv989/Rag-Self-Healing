import os
import chromadb
from langchain_chroma import Chroma
from src.splitter import split_documents

CHROMA_DIR = "chroma_db"
DATA_DIR = "Data"


def _get_embeddings():
    """
    Returns embedding model:
    - If GROQ_API_KEY or EMBEDDING_PROVIDER == 'huggingface' or Ollama is unavailable, uses HuggingFaceEmbeddings.
    - Otherwise uses local OllamaEmbeddings(model='mxbai-embed-large').
    """
    provider = os.environ.get("EMBEDDING_PROVIDER", "").lower()
    has_groq = bool(os.environ.get("GROQ_API_KEY"))

    if provider == "huggingface" or has_groq:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large")
        except Exception as e:
            print(f"[embedder] HuggingFaceEmbeddings fallback error: {e}")

    # Default to Ollama with automatic HuggingFace fallback
    try:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model="mxbai-embed-large")
    except Exception:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large")


def create_vector_db(pdf_path):
    """Ingest a document and add its chunks into the shared ChromaDB collection."""
    chunks = split_documents(pdf_path)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=_get_embeddings(),
        persist_directory=CHROMA_DIR
    )

    return vectorstore


def list_ingested_documents():
    """
    Returns a list of ingested documents with their chunk counts.
    Derives document list directly from ChromaDB collection metadata (source field).
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name="langchain")
        db = collection.get(include=["metadatas"])
        metadatas = db.get("metadatas", []) or []

        doc_chunks = {}
        for meta in metadatas:
            if not meta:
                continue
            source = meta.get("source", "Unknown")
            name = source.replace("\\", "/").split("/")[-1]
            doc_chunks[name] = doc_chunks.get(name, 0) + 1

        result = []
        for name, count in doc_chunks.items():
            file_path = os.path.join(DATA_DIR, name)
            if not os.path.exists(file_path):
                file_path = os.path.join("data", name)
            
            size = os.path.getsize(file_path) if os.path.exists(file_path) else None
            result.append({
                "name": name,
                "chunk_count": count,
                "size_bytes": size
            })

        return result
    except Exception as e:
        print(f"Error in list_ingested_documents: {e}")
        return []


def delete_document_collection(doc_name: str) -> bool:
    """
    Delete all chunks belonging to a specific document from ChromaDB.
    Returns True if any chunks were deleted, False if none found.
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name="langchain")
        db = collection.get(include=["metadatas"])
        ids = db.get("ids", []) or []
        metadatas = db.get("metadatas", []) or []

        ids_to_delete = []
        for chunk_id, meta in zip(ids, metadatas):
            if not meta:
                continue
            source = meta.get("source", "")
            name = source.replace("\\", "/").split("/")[-1]
            if name == doc_name:
                ids_to_delete.append(chunk_id)

        if not ids_to_delete:
            return False

        collection.delete(ids=ids_to_delete)

        for folder in [DATA_DIR, "data"]:
            file_path = os.path.join(folder, doc_name)
            if os.path.exists(file_path):
                os.remove(file_path)

        return True
    except Exception as e:
        print(f"Error in delete_document_collection: {e}")
        return False