from rank_bm25 import BM25Okapi
import os
from src.embedder import _get_embeddings, get_or_create_vectorstore

import src.state as state
from src.reranker import Reranker
from src.confidence_checker import ConfidenceChecker
from src.healing_policy import HealingPolicy
from src.reflection_agent import ReflectionAgent
from src.failure_logger import log_failure
from src.failure_analytics import pretty_failure_report
from src.adaptive_healing import adaptive_decision

from src.memory import store_mentions, store_valid_entities, get_latest_entity, memory_relevant
from src.query_processor import resolve_query, is_ambiguous_multi_entity
from src.retrieval import retrieve_documents
from src.generation import generate_memory_answer_stream, generate_answer_stream, correct_answer_stream
import json
import asyncio

MAX_HEAL_ATTEMPTS = 2


def initialize_system():
    embeddings = _get_embeddings()

    if state.vectorstore is None:
        state.vectorstore = get_or_create_vectorstore()
        state.cache_store = get_or_create_vectorstore()

        try:
            db = state.vectorstore.get()
            state.ALL_DOCS = db.get("documents", []) or []
            state.ALL_METADATA = db.get("metadatas", []) or []
        except Exception as e:
            print(f"Error loading Vector DB: {e}")
            state.ALL_DOCS = []
            state.ALL_METADATA = []

        if state.ALL_DOCS:
            tokenized_corpus = [
                doc.lower().split()
                for doc in state.ALL_DOCS
            ]
            state.bm25 = BM25Okapi(tokenized_corpus)
        else:
            state.bm25 = None

    if state.llm is None:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if groq_api_key:
            try:
                from langchain_groq import ChatGroq
                state.llm = ChatGroq(
                    model_name=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                    groq_api_key=groq_api_key,
                    temperature=0
                )
                print("[rag_engine] LLM initialized using Groq API (Cloud Mode).")
            except Exception as e:
                print(f"[rag_engine] ChatGroq init error: {e}. Falling back to ChatOllama.")
                from langchain_ollama import ChatOllama
                state.llm = ChatOllama(model="llama3.2", temperature=0)
        else:
            from langchain_ollama import ChatOllama
            state.llm = ChatOllama(
                model="llama3.2",
                temperature=0
            )
            print("[rag_engine] LLM initialized using local Ollama (Local Mode).")

    if state.reflection_agent is None:
        state.reflection_agent = ReflectionAgent(state.llm)
    if state.reranker is None:
        state.reranker = Reranker()
    if state.confidence_checker is None:
        state.confidence_checker = ConfidenceChecker()
    if state.healing_policy is None:
        state.healing_policy = HealingPolicy()


async def ask_question_stream(query):
    initialize_system()

    if query.lower().strip() == "analyze failures":
        yield f"data: {{'type': 'metadata', 'strategy': 'REPORT', 'heals': 0, 'confidence': 1.0, 'sources': []}}\n\n".replace("'", '"')
        yield "data: " + json.dumps({'type': 'chunk', 'text': pretty_failure_report()}) + "\n\n"
        yield "data: [DONE]\n\n"
        return

    from src.semantic_cache import check_cache, store_in_cache
    cache_hit = check_cache(query)
    if cache_hit:
        yield "data: " + json.dumps({'type': 'metadata', 'strategy': 'CACHE', 'heals': 0, 'confidence': 1.0, 'sources': cache_hit['sources']}) + "\n\n"
        yield "data: " + json.dumps({'type': 'chunk', 'text': cache_hit['answer']}) + "\n\n"
        yield "data: [DONE]\n\n"
        return

    store_mentions(query)
    
    from src.query_processor import extract_entities
    if extract_entities(query):
        store_valid_entities(query)

    search_query = resolve_query(query, get_latest_entity())

    if search_query is None:
        log_failure(query, query, "CLARIFY", [], "pronoun_without_entity")
        yield "data: " + json.dumps({'type': 'metadata', 'strategy': 'REFUSE', 'heals': 0, 'confidence': 0.0, 'sources': []}) + "\n\n"
        yield "data: " + json.dumps({'type': 'chunk', 'text': 'Who are you referring to?'}) + "\n\n"
        yield "data: [DONE]\n\n"
        return

    if is_ambiguous_multi_entity(query, state.mention_memory["recent_mentions"]):
        mentions = state.mention_memory["recent_mentions"][-2:]
        yield "data: " + json.dumps({'type': 'metadata', 'strategy': 'CLARIFY', 'heals': 0, 'confidence': 0.0, 'sources': []}) + "\n\n"
        yield "data: " + json.dumps({'type': 'chunk', 'text': f'Do you mean between {mentions[0]} and {mentions[1]}?'}) + "\n\n"
        yield "data: [DONE]\n\n"
        return

    if memory_relevant(search_query):
        yield "data: " + json.dumps({'type': 'metadata', 'strategy': 'MEMORY', 'heals': 0, 'confidence': 1.0, 'sources': ['Conversation Memory']}) + "\n\n"
        memory_answer_text = ""
        async for chunk in generate_memory_answer_stream(search_query):
            memory_answer_text += chunk
            yield "data: " + json.dumps({'type': 'chunk', 'text': chunk}) + "\n\n"
        if memory_answer_text:
            yield "data: [DONE]\n\n"
            return

    docs = await retrieve_documents(search_query)

    if not docs:
        yield "data: " + json.dumps({'type': 'metadata', 'strategy': 'REFUSE', 'heals': 0, 'confidence': 0.0, 'sources': []}) + "\n\n"
        yield "data: " + json.dumps({'type': 'chunk', 'text': 'No relevant documents found.'}) + "\n\n"
        yield "data: [DONE]\n\n"
        return

    ranked_docs = state.reranker.rerank(search_query, docs, top_k=3)

    low_confidence = not state.confidence_checker.is_confident(ranked_docs)
    heal_attempts = 0
    strategy = "RAG"

    while low_confidence and heal_attempts < MAX_HEAL_ATTEMPTS:
        heal_attempts += 1

        base_strategy = state.healing_policy.decide(
            ranked_docs,
            query,
            search_query,
            get_latest_entity()
        )

        strategy = adaptive_decision(base_strategy)

        if strategy == "REFUSE":
            yield "data: " + json.dumps({'type': 'metadata', 'strategy': strategy, 'heals': heal_attempts, 'confidence': 0.0, 'sources': []}) + "\n\n"
            yield "data: " + json.dumps({'type': 'chunk', 'text': 'I could not confidently find relevant information.'}) + "\n\n"
            yield "data: [DONE]\n\n"
            return

        if strategy == "MMR":
            mmr_docs = await retrieve_documents(search_query, "mmr")
            ranked_docs = state.reranker.rerank(search_query, mmr_docs, top_k=3)
            low_confidence = not state.confidence_checker.is_confident(ranked_docs, retry=True)

    top_docs = [doc for doc, _ in ranked_docs]
    context = "\n\n".join(doc.page_content for doc in top_docs)
    
    confidence = ranked_docs[0][1] if ranked_docs else 0.0
    if hasattr(confidence, "item"):
        confidence = float(confidence.item())

    sources = []
    for doc in top_docs:
        source_name = doc.metadata.get("source", "Unknown Source")
        if "\\" in source_name or "/" in source_name:
            source_name = source_name.replace("\\", "/").split("/")[-1]
        page = doc.metadata.get("page")
        if page is not None:
            sources.append(f"{source_name} (Page {page})")
        else:
            sources.append(source_name)
    unique_sources = list(dict.fromkeys(sources))

    yield "data: " + json.dumps({'type': 'metadata', 'strategy': strategy, 'heals': heal_attempts, 'confidence': confidence, 'sources': unique_sources}) + "\n\n"

    draft_chunks = []
    async for chunk in generate_answer_stream(context, query):
        draft_chunks.append(chunk)
        yield "data: " + json.dumps({'type': 'chunk', 'text': chunk}) + "\n\n"
        
    draft_answer = "".join(draft_chunks)

    decision = await state.reflection_agent.decide_async(
        query=query,
        context=context,
        draft_answer=draft_answer,
        current_entity=get_latest_entity()
    )

    if decision == "REGENERATE":
        yield "data: " + json.dumps({'type': 'chunk', 'text': '\n\n**Self-Correction Triggered:**\n\n'}) + "\n\n"
        final_chunks = []
        async for chunk in correct_answer_stream(context, query, draft_answer):
            final_chunks.append(chunk)
            yield "data: " + json.dumps({'type': 'chunk', 'text': chunk}) + "\n\n"
        final_answer = "".join(final_chunks)
    else:
        final_answer = draft_answer

    state.chat_history.append({
        "user": query,
        "assistant": final_answer
    })
    state.save_session()

    store_in_cache(query, final_answer, unique_sources)

    yield "data: [DONE]\n\n"
