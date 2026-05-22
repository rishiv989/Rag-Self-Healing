import json
import src.state as state
from langchain_core.documents import Document

# Threshold for semantic similarity. Lower means more strictly similar.
# Adjust depending on embedding model (L2 distance or Cosine distance default in Chroma)
CACHE_THRESHOLD = 0.15 

def check_cache(query):
    """
    Checks if a semantically similar query was asked recently.
    """
    if not hasattr(state, "cache_store") or state.cache_store is None:
        return None
    
    try:
        results = state.cache_store.similarity_search_with_score(query, k=1)
        if results:
            doc, distance = results[0]
            # Chroma default is L2 distance, lower is closer
            if distance <= CACHE_THRESHOLD:
                return {
                    "answer": doc.metadata.get("answer"),
                    "sources": json.loads(doc.metadata.get("sources", "[]")),
                    "distance": float(distance)
                }
    except Exception as e:
        print(f"Cache check error: {e}")
        
    return None

def store_in_cache(query, answer, sources):
    """
    Stores the query and the generated answer into the semantic cache.
    """
    if not hasattr(state, "cache_store") or state.cache_store is None:
        return
    
    try:
        doc = Document(
            page_content=query,
            metadata={
                "answer": answer,
                "sources": json.dumps(sources)
            }
        )
        state.cache_store.add_documents([doc])
    except Exception as e:
        print(f"Cache store error: {e}")
