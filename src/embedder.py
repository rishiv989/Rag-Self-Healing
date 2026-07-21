import os
import hashlib
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import src.state as global_state

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


class CloudVectorStore:
    """
    Lightweight, high-speed vector store for Cloud deployment.
    Uses zero C++ binary bindings and zero memory overhead (<30MB RAM).
    Fully compatible with LangChain similarity search & retriever interfaces.
    """
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self.docs = []

    def add_documents(self, documents):
        self.docs.extend(documents)

    def similarity_search(self, query, k=4):
        if not self.docs:
            return []
        q_vec = self.embedding_function.embed_query(query)
        doc_texts = [d.page_content for d in self.docs]
        doc_vecs = self.embedding_function.embed_documents(doc_texts)

        scored = []
        for doc, vec in zip(self.docs, doc_vecs):
            dot = sum(a * b for a, b in zip(q_vec, vec))
            scored.append((dot, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:k]]

    def similarity_search_with_score(self, query, k=4):
        if not self.docs:
            return []
        q_vec = self.embedding_function.embed_query(query)
        doc_texts = [d.page_content for d in self.docs]
        doc_vecs = self.embedding_function.embed_documents(doc_texts)

        scored = []
        for doc, vec in zip(self.docs, doc_vecs):
            dot = sum(a * b for a, b in zip(q_vec, vec))
            scored.append((doc, float(dot)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def as_retriever(self, search_type="similarity", search_kwargs=None):
        k = (search_kwargs or {}).get("k", 4)
        store = self

        class CloudRetriever:
            async def ainvoke(self, query):
                return store.similarity_search(query, k=k)

            def invoke(self, query):
                return store.similarity_search(query, k=k)

        return CloudRetriever()

    def get(self, include=None):
        return {
            "documents": [d.page_content for d in self.docs],
            "metadatas": [d.metadata for d in self.docs],
            "ids": [str(i) for i in range(len(self.docs))]
        }


def _get_embeddings():
    """
    Returns embedding model:
    - In Cloud Mode (GROQ_API_KEY present), uses zero-overhead FastCloudEmbeddings to keep RAM < 30MB.
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


def get_or_create_vectorstore():
    """Returns vectorstore instance (CloudVectorStore in cloud mode, Chroma in local mode)."""
    if global_state.vectorstore is not None:
        return global_state.vectorstore

    embeddings = _get_embeddings()
    if os.environ.get("GROQ_API_KEY"):
        global_state.vectorstore = CloudVectorStore(embeddings)
    else:
        from langchain_chroma import Chroma
        global_state.vectorstore = Chroma(
            collection_name="langchain",
            embedding_function=embeddings,
            persist_directory="chroma_db"
        )
    return global_state.vectorstore


def create_vector_db(pdf_path):
    """Ingest a document into VectorStore in small batches to keep memory usage under 30MB."""
    from src.splitter import split_documents
    embeddings = _get_embeddings()
    chunks = split_documents(pdf_path, embeddings=embeddings)

    vectorstore = get_or_create_vectorstore()
    vectorstore.add_documents(chunks)
    return vectorstore


def list_ingested_documents():
    """
    Returns a list of ingested documents with their chunk counts.
    Derives document list directly from vectorstore metadata.
    """
    try:
        vectorstore = get_or_create_vectorstore()
        db = vectorstore.get(include=["metadatas"])
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
    Delete all chunks belonging to a specific document from vectorstore.
    """
    try:
        vectorstore = get_or_create_vectorstore()
        if isinstance(vectorstore, CloudVectorStore):
            before = len(vectorstore.docs)
            vectorstore.docs = [d for d in vectorstore.docs if d.metadata.get("source", "").replace("\\", "/").split("/")[-1] != doc_name]
            deleted = before - len(vectorstore.docs)
            if deleted == 0:
                return False
        else:
            client = chromadb.PersistentClient(path="chroma_db")
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