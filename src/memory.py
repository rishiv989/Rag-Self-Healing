from langchain_core.documents import Document
import src.state as state
from src.query_processor import extract_entities, is_broad_query

MEMORY_RELEVANCE_THRESHOLD = 0.5


def store_mentions(query):
    entities = extract_entities(query)

    for entity in entities:
        if entity in state.mention_memory["recent_mentions"]:
            state.mention_memory["recent_mentions"].remove(entity)

        state.mention_memory["recent_mentions"].append(entity)

    state.mention_memory["recent_mentions"] = state.mention_memory["recent_mentions"][-5:]
    state.save_session()


def store_valid_entities(query):
    entities = extract_entities(query)

    for entity in entities:
        if entity in state.entity_memory["recent_entities"]:
            state.entity_memory["recent_entities"].remove(entity)

        state.entity_memory["recent_entities"].append(entity)

    state.entity_memory["recent_entities"] = state.entity_memory["recent_entities"][-5:]
    state.save_session()


def get_latest_entity():
    entities = state.entity_memory["recent_entities"]

    if not entities:
        return None

    return entities[-1]


def memory_relevant(query):
    if not state.chat_history:
        return False

    if is_broad_query(query):
        return False

    # Only use memory when the query explicitly references a pronoun (he/she/it/they)
    # OR a known entity that was previously mentioned.
    # This prevents the cloud reranker's constant score=1.0 from triggering memory
    # on generic knowledge questions like "what is software development?".
    from src.query_processor import has_pronoun, extract_entities
    known_entities = [e.lower() for e in state.entity_memory.get("recent_entities", [])]

    has_reference = has_pronoun(query)
    if not has_reference and known_entities:
        query_lower = query.lower()
        has_reference = any(entity in query_lower for entity in known_entities)

    if not has_reference:
        return False

    latest_answer = state.chat_history[-1]["assistant"]

    pseudo_doc = Document(
        page_content=latest_answer,
        metadata={"source": "memory"}
    )

    ranked = state.reranker.rerank(query, [pseudo_doc], top_k=1)

    if not ranked:
        return False

    _, score = ranked[0]
    return score >= MEMORY_RELEVANCE_THRESHOLD
