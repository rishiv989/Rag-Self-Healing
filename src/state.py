from langchain_core.documents import Document
from typing import List, Dict
import json
import os

SESSION_FILE = "Data/user_session.json"

def load_session():
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {
        "chat_history": [],
        "entity_memory": {"recent_entities": []},
        "mention_memory": {"recent_mentions": []}
    }

session_data = load_session()

# Shared runtime state
chat_history: List[Dict[str, str]] = session_data["chat_history"]
entity_memory = session_data["entity_memory"]
mention_memory = session_data["mention_memory"]

def save_session():
    os.makedirs(os.path.dirname(SESSION_FILE), exist_ok=True)
    with open(SESSION_FILE, "w") as f:
        json.dump({
            "chat_history": chat_history,
            "entity_memory": entity_memory,
            "mention_memory": mention_memory
        }, f)

# System models
llm = None
vectorstore = None
cache_store = None
bm25 = None
reflection_agent = None
reranker = None
confidence_checker = None
healing_policy = None

ALL_DOCS = []
ALL_METADATA = []