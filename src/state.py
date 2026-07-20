from langchain_core.documents import Document
from typing import List, Dict
import json
import os
import sqlite3

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "Data"
SESSION_FILE = os.path.join(DATA_DIR, "user_session.json")
DB_FILE = os.path.join(DATA_DIR, "memory.db")

# ─────────────────────────────────────────────────────────────────────────────
# SQLite Chat History (persistent across restarts)
# ─────────────────────────────────────────────────────────────────────────────

def _get_db_connection():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    """Create tables if they don't exist."""
    with _get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT NOT NULL,
                assistant_message TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def load_chat_history_from_db(limit: int = 50) -> List[Dict[str, str]]:
    """Load the last `limit` messages from SQLite."""
    try:
        _init_db()
        with _get_db_connection() as conn:
            rows = conn.execute(
                "SELECT user_message, assistant_message FROM chat_history ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        # Return in chronological order
        return [{"user": row["user_message"], "assistant": row["assistant_message"]} for row in reversed(rows)]
    except Exception:
        return []


def save_message_to_db(user: str, assistant: str):
    """Persist a single conversation turn to SQLite."""
    try:
        _init_db()
        with _get_db_connection() as conn:
            conn.execute(
                "INSERT INTO chat_history (user_message, assistant_message) VALUES (?, ?)",
                (user, assistant)
            )
            conn.commit()
    except Exception as e:
        print(f"[state] DB write error: {e}")


def clear_chat_history_db():
    """Wipe all chat history from SQLite."""
    try:
        _init_db()
        with _get_db_connection() as conn:
            conn.execute("DELETE FROM chat_history")
            conn.commit()
    except Exception as e:
        print(f"[state] DB clear error: {e}")


def get_full_chat_history_from_db() -> List[Dict[str, str]]:
    """Return the complete chat history (for the /chat/history endpoint)."""
    try:
        _init_db()
        with _get_db_connection() as conn:
            rows = conn.execute(
                "SELECT user_message, assistant_message, created_at FROM chat_history ORDER BY id"
            ).fetchall()
        return [
            {"user": row["user_message"], "assistant": row["assistant_message"], "timestamp": row["created_at"]}
            for row in rows
        ]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# JSON Session (entity + mention memory — still file-based, it's small)
# ─────────────────────────────────────────────────────────────────────────────

def load_session():
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "entity_memory": {"recent_entities": []},
        "mention_memory": {"recent_mentions": []}
    }


session_data = load_session()

# In-memory chat history (kept in sync with SQLite)
chat_history: List[Dict[str, str]] = load_chat_history_from_db()
entity_memory = session_data.get("entity_memory", {"recent_entities": []})
mention_memory = session_data.get("mention_memory", {"recent_mentions": []})


def save_session():
    """
    Persist entity and mention memory to JSON.
    Chat history is persisted immediately via save_message_to_db().
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SESSION_FILE, "w") as f:
        json.dump({
            "entity_memory": entity_memory,
            "mention_memory": mention_memory
        }, f)


def save_message(user: str, assistant: str):
    """
    Save a conversation turn:
      - Appends to in-memory chat_history
      - Persists to SQLite
      - Saves entity/mention memory to JSON
    """
    chat_history.append({"user": user, "assistant": assistant})
    save_message_to_db(user, assistant)
    save_session()


# ─────────────────────────────────────────────────────────────────────────────
# System Models (set by rag_engine.py at startup)
# ─────────────────────────────────────────────────────────────────────────────

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