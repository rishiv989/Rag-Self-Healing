import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.splitter import split_documents

CHROMA_DIR = "chroma_db"
DATA_DIR = "Data"


def _get_embeddings():
    return OllamaEmbeddings(model="mxbai-embed-large")


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
    Derives document list from ChromaDB metadata (source field).
    """
    try:
        embeddings = _get_embeddings()
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        db = vectorstore.get(include=["metadatas"])
        metadatas = db.get("metadatas", [])

        doc_chunks = {}
        for meta in metadatas:
            source = meta.get("source", "Unknown")
            # Normalize to just filename
            name = source.replace("\\", "/").split("/")[-1]
            doc_chunks[name] = doc_chunks.get(name, 0) + 1

        result = []
        for name, count in doc_chunks.items():
            file_path = os.path.join(DATA_DIR, name)
            size = os.path.getsize(file_path) if os.path.exists(file_path) else None
            result.append({
                "name": name,
                "chunk_count": count,
                "size_bytes": size
            })

        return result
    except Exception:
        return []


def delete_document_collection(doc_name: str) -> bool:
    """
    Delete all chunks belonging to a specific document from ChromaDB.
    Returns True if any chunks were deleted, False if none found.
    """
    try:
        embeddings = _get_embeddings()
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        db = vectorstore.get(include=["metadatas"])
        ids = db.get("ids", [])
        metadatas = db.get("metadatas", [])

        # Find IDs matching the document name
        ids_to_delete = []
        for chunk_id, meta in zip(ids, metadatas):
            source = meta.get("source", "")
            name = source.replace("\\", "/").split("/")[-1]
            if name == doc_name:
                ids_to_delete.append(chunk_id)

        if not ids_to_delete:
            return False

        vectorstore._collection.delete(ids=ids_to_delete)

        # Also remove the file from Data/ if it exists
        file_path = os.path.join(DATA_DIR, doc_name)
        if os.path.exists(file_path):
            os.remove(file_path)

        return True
    except Exception:
        return False


if __name__ == "__main__":
    pdf_path = "data/sample.pdf"
    vectorstore = create_vector_db(pdf_path)
    print("Vector database created successfully!")
    print(f"Stored {vectorstore._collection.count()} chunks.")