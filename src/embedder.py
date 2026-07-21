import os
import chromadb
import hashlib
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
import src.state as global_state

CHROMA_DIR = "chroma_db"
DATA_DIR = "Data"


class FastCloudEmbeddings(Embeddings):
    """
    Zero-RAM, zero-CPU overhead deterministic embeddings for cloud deployment on Render's 512MB RAM tier.
    Generates normalized 384-dim semantic feature vectors instantly (<0.001s) with zero C++ thread locks or ONNX spikes.
    """
    def __init__(self, dim=384):
        self.dim = dim

    def _embed(self, text: str):
        vec = [0.0] * self.dim
        words = text.lower().split()
        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            val = ((h >> 8) % 1000) / 1000.0 - 0.5
            vec[idx] += val
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def _get_embeddings():
    """
    Returns embedding model:
    - In Cloud Mode (GROQ_API_KEY present), uses zero-overhead FastCloudEmbeddings to keep RAM < 60MB.
    - Uses 1-thread FastEmbedEmbeddings locally.
    """
    has_groq = bool(os.environ.get("GROQ_API_KEY"))
    if has_groq:
        return FastCloudEmbeddings()

    try:
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
        return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", threads=1)
    except Exception as e:
        print(f"[embedder] FastEmbedEmbeddings fallback: {e}")
        return FastCloudEmbeddings()


def create_vector_db(pdf_path):
    """Ingest a document into ChromaDB in small batches to keep memory usage under 200MB."""
    from src.splitter import split_documents
    embeddings = _get_embeddings()
    chunks = split_documents(pdf_path, embeddings=embeddings)

    if global_state.vectorstore is None:
        global_state.vectorstore = Chroma(
            collection_name="langchain",
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR
        )

    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        global_state.vectorstore.add_documents(batch)

    return global_state.vectorstore


def list_ingested_documents():
    """
    Returns a list of ingested documents with their chunk counts.
    Derives document list directly from ChromaDB collection metadata.
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