from typing import TypedDict, List, Any
import asyncio
import json
import os
from langgraph.graph import StateGraph, END

import src.state as global_state
from src.memory import store_mentions, store_valid_entities, get_latest_entity, memory_relevant
from src.query_processor import resolve_query, extract_entities, is_ambiguous_multi_entity
from src.retrieval import retrieve_documents
from src.semantic_cache import check_cache, store_in_cache
from src.generation import generate_memory_answer_stream, generate_answer_stream, correct_answer_stream
from src.failure_logger import log_failure
from src.failure_analytics import pretty_failure_report
from src.adaptive_healing import adaptive_decision
from ddgs import DDGS

MAX_HEAL_ATTEMPTS = 2

class AgentState(TypedDict):
    query: str
    search_query: str
    docs: List[Any]
    ranked_docs: List[Any]
    context: str
    graph_context: str
    strategy: str
    heal_attempts: int
    confidence: float
    sources: List[str]
    draft_answer: str
    final_answer: str
    queue: asyncio.Queue
    is_terminal: bool

async def generate_sse(queue: asyncio.Queue, type_str: str, **kwargs):
    kwargs["type"] = type_str
    data = json.dumps(kwargs)
    await queue.put(f"data: {data}\n\n")

# --- NODES ---

async def analyze_query(state: AgentState):
    query = state["query"]
    queue = state["queue"]
    await generate_sse(queue, "node", current_node="analyze_query")

    if query.lower().strip() == "analyze failures":
        await generate_sse(queue, "metadata", strategy="REPORT", heals=0, confidence=1.0, sources=[])
        await generate_sse(queue, "chunk", text=pretty_failure_report())
        await queue.put("data: [DONE]\n\n")
        return {"is_terminal": True}

    store_mentions(query)
    if extract_entities(query):
        store_valid_entities(query)

    search_query = resolve_query(query, get_latest_entity())

    if search_query is None:
        log_failure(query, query, "CLARIFY", [], "pronoun_without_entity")
        await generate_sse(queue, "metadata", strategy="REFUSE", heals=0, confidence=0.0, sources=[])
        await generate_sse(queue, "chunk", text="Who are you referring to?")
        await queue.put("data: [DONE]\n\n")
        return {"is_terminal": True}

    if is_ambiguous_multi_entity(query, global_state.mention_memory["recent_mentions"]):
        mentions = global_state.mention_memory["recent_mentions"][-2:]
        await generate_sse(queue, "metadata", strategy="CLARIFY", heals=0, confidence=0.0, sources=[])
        await generate_sse(queue, "chunk", text=f"Do you mean between {mentions[0]} and {mentions[1]}?")
        await queue.put("data: [DONE]\n\n")
        return {"is_terminal": True}

    return {"search_query": search_query, "is_terminal": False}

async def check_fast_memory(state: AgentState):
    query = state["query"]
    search_query = state["search_query"]
    queue = state["queue"]
    await generate_sse(queue, "node", current_node="check_fast_memory")

    cache_hit = check_cache(query)
    if cache_hit:
        await generate_sse(queue, "metadata", strategy="CACHE", heals=0, confidence=1.0, sources=cache_hit['sources'])
        await generate_sse(queue, "chunk", text=cache_hit['answer'])
        await queue.put("data: [DONE]\n\n")
        return {"is_terminal": True}

    if memory_relevant(search_query):
        await generate_sse(queue, "metadata", strategy="MEMORY", heals=0, confidence=1.0, sources=["Conversation Memory"])
        ans = ""
        async for chunk in generate_memory_answer_stream(search_query):
            ans += chunk
            await generate_sse(queue, "chunk", text=chunk)
        if ans:
            await queue.put("data: [DONE]\n\n")
            return {"is_terminal": True}

    return {"is_terminal": False}

async def retrieve_and_rerank(state: AgentState):
    search_query = state["search_query"]
    queue = state["queue"]
    await generate_sse(queue, "node", current_node="retrieve_and_rerank")

    docs = await retrieve_documents(search_query)

    if not docs:
        return {"heal_attempts": 0, "strategy": "WEB_SEARCH", "is_terminal": False}

    ranked_docs = global_state.reranker.rerank(search_query, docs, top_k=3)
    confidence = float(ranked_docs[0][1]) if ranked_docs else 0.0

    return {
        "docs": docs, 
        "ranked_docs": ranked_docs, 
        "confidence": confidence,
        "heal_attempts": 0,
        "strategy": "RAG",
        "is_terminal": False
    }

async def self_heal(state: AgentState):
    search_query = state["search_query"]
    query = state["query"]
    ranked_docs = state["ranked_docs"]
    heal_attempts = state["heal_attempts"]
    queue = state["queue"]
    await generate_sse(queue, "node", current_node="self_heal")

    is_conf = global_state.confidence_checker.is_confident(ranked_docs, retry=(heal_attempts > 0))
    
    if is_conf or heal_attempts >= MAX_HEAL_ATTEMPTS:
        return {}

    heal_attempts += 1
    base_strategy = global_state.healing_policy.decide(ranked_docs, query, search_query, get_latest_entity())
    strategy = adaptive_decision(base_strategy)

    if strategy == "REFUSE":
        return {"heal_attempts": heal_attempts, "strategy": "WEB_SEARCH", "is_terminal": False}

    if strategy == "MMR":
        mmr_docs = await retrieve_documents(search_query, "mmr")
        ranked_docs = global_state.reranker.rerank(search_query, mmr_docs, top_k=3)
        confidence = float(ranked_docs[0][1]) if ranked_docs else 0.0
        return {
            "ranked_docs": ranked_docs,
            "heal_attempts": heal_attempts,
            "strategy": strategy,
            "confidence": confidence
        }

    return {"heal_attempts": heal_attempts, "strategy": strategy}


async def build_knowledge_graph(state: AgentState):
    ranked_docs = state["ranked_docs"]
    queue = state["queue"]
    
    await generate_sse(queue, "node", current_node="build_knowledge_graph")
    
    if os.environ.get("GROQ_API_KEY"):
        return {"graph_context": ""}

    await generate_sse(queue, "chunk", text="\n\n*[GraphRAG] Extracting Knowledge Graph...*\n")
    
    top_docs = [doc for doc, _ in ranked_docs]
    context = "\n\n".join(doc.page_content for doc in top_docs)
    
    prompt = f"""
Extract the most important entity relationships from the text below.
Output ONLY a list of relationships in this exact format:
[Entity 1] -> [Relationship] -> [Entity 2]

Text:
{context}
"""
    
    graph_context = ""
    try:
        async for chunk in global_state.llm.astream(prompt):
            graph_context += chunk.content
        await generate_sse(queue, "chunk", text=graph_context + "\n\n")
    except Exception as e:
        print(f"[graph] build_knowledge_graph error: {e}")
        graph_context = ""

    return {"graph_context": graph_context}

async def generate_draft(state: AgentState):
    ranked_docs = state["ranked_docs"]
    query = state["query"]
    queue = state["queue"]
    graph_context = state.get("graph_context", "")
    await generate_sse(queue, "node", current_node="generate_draft")
    
    top_docs = [doc for doc, _ in ranked_docs]
    context = "\n\n".join(doc.page_content for doc in top_docs)
    
    if graph_context:
        context = f"KNOWLEDGE GRAPH:\n{graph_context}\n\nDOCUMENT CONTEXT:\n{context}"

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

    await generate_sse(queue, "metadata", strategy=state.get("strategy", "RAG"), heals=state.get("heal_attempts", 0), confidence=state.get("confidence", 1.0), sources=unique_sources)

    draft_chunks = []
    async for chunk in generate_answer_stream(context, query):
        draft_chunks.append(chunk)
        await generate_sse(queue, "chunk", text=chunk)
        
    draft_answer = "".join(draft_chunks)

    return {"context": context, "draft_answer": draft_answer, "sources": unique_sources}

async def reflect(state: AgentState):
    query = state["query"]
    context = state["context"]
    draft_answer = state["draft_answer"]
    queue = state["queue"]
    await generate_sse(queue, "node", current_node="reflect")

    try:
        decision = await asyncio.to_thread(
            global_state.reflection_agent.decide,
            query=query,
            context=context,
            draft_answer=draft_answer,
            current_entity=get_latest_entity()
        )
    except Exception as e:
        print(f"[graph] reflect error: {e}")
        decision = "ACCEPT"

    if decision == "REGENERATE":
        await generate_sse(queue, "chunk", text="\n\n**Self-Correction Triggered:**\n\n")
        final_chunks = []
        async for chunk in correct_answer_stream(context, query, draft_answer):
            final_chunks.append(chunk)
            await generate_sse(queue, "chunk", text=chunk)
        final_answer = "".join(final_chunks)
    else:
        final_answer = draft_answer

    global_state.save_message(query, final_answer)
    store_in_cache(query, final_answer, state.get("sources", []))

    await queue.put("data: [DONE]\n\n")
    return {"final_answer": final_answer, "is_terminal": True}

async def web_search_fallback(state: AgentState):
    query = state["query"]
    queue = state["queue"]
    heal_attempts = state.get("heal_attempts", 0)
    await generate_sse(queue, "node", current_node="web_search_fallback")

    await generate_sse(queue, "metadata", strategy="WEB_SEARCH", heals=heal_attempts, confidence=0.0, sources=["DuckDuckGo Search"])
    await generate_sse(queue, "chunk", text="*Searching the live web...*\n\n")

    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            await generate_sse(queue, "chunk", text="I could not find any relevant information in your documents or on the web.")
            await queue.put("data: [DONE]\n\n")
            return {"is_terminal": True}

        context = "\n\n".join([f"Source: {res['href']}\nSnippet: {res['body']}" for res in results])
        sources = [res['href'] for res in results]
        
        draft_chunks = []
        async for chunk in generate_answer_stream(context, query):
            draft_chunks.append(chunk)
            await generate_sse(queue, "chunk", text=chunk)
            
        final_answer = "".join(draft_chunks)
        global_state.save_message(query, final_answer)
        store_in_cache(query, final_answer, sources)
        
    except Exception as e:
        await generate_sse(queue, "chunk", text=f"Web search failed: {str(e)}")
        
    await queue.put("data: [DONE]\n\n")
    return {"is_terminal": True}


# --- EDGES ---

def route_analyze(state: AgentState):
    return END if state.get("is_terminal") else "check_fast_memory"

def route_fast_memory(state: AgentState):
    return END if state.get("is_terminal") else "retrieve_and_rerank"

def route_retrieve(state: AgentState):
    if state.get("is_terminal"):
        return END
    if state.get("strategy") == "WEB_SEARCH":
        return "web_search_fallback"
    return "self_heal"

def route_heal(state: AgentState):
    if state.get("is_terminal"):
        return END
    
    if state.get("strategy") == "WEB_SEARCH":
        return "web_search_fallback"
    
    is_conf = global_state.confidence_checker.is_confident(state.get("ranked_docs", []), retry=(state.get("heal_attempts", 0) > 0))
    if is_conf or state.get("heal_attempts", 0) >= MAX_HEAL_ATTEMPTS:
        return "build_knowledge_graph"
    
    return "self_heal"

# Build Graph
builder = StateGraph(AgentState)
builder.add_node("analyze_query", analyze_query)
builder.add_node("check_fast_memory", check_fast_memory)
builder.add_node("retrieve_and_rerank", retrieve_and_rerank)
builder.add_node("self_heal", self_heal)
builder.add_node("build_knowledge_graph", build_knowledge_graph)
builder.add_node("generate_draft", generate_draft)
builder.add_node("reflect", reflect)
builder.add_node("web_search_fallback", web_search_fallback)

builder.set_entry_point("analyze_query")
builder.add_conditional_edges("analyze_query", route_analyze)
builder.add_conditional_edges("check_fast_memory", route_fast_memory)
builder.add_conditional_edges("retrieve_and_rerank", route_retrieve)
builder.add_conditional_edges("self_heal", route_heal)
builder.add_edge("build_knowledge_graph", "generate_draft")
builder.add_edge("generate_draft", "reflect")
builder.add_edge("reflect", END)
builder.add_edge("web_search_fallback", END)

rag_graph = builder.compile()

async def run_langgraph_stream(query: str):
    if global_state.llm is None or global_state.vectorstore is None:
        from src.rag_engine import initialize_system
        initialize_system()

    queue = asyncio.Queue()
    
    async def process_graph():
        try:
            await rag_graph.ainvoke({
                "query": query,
                "search_query": "",
                "docs": [],
                "ranked_docs": [],
                "context": "",
                "graph_context": "",
                "strategy": "",
                "heal_attempts": 0,
                "confidence": 0.0,
                "sources": [],
                "draft_answer": "",
                "final_answer": "",
                "queue": queue,
                "is_terminal": False
            })
        except Exception as e:
            print(f"Graph execution error: {e}")
            await generate_sse(queue, "chunk", text=f"Error: {str(e)}")
            await queue.put("data: [DONE]\n\n")

    task = asyncio.create_task(process_graph())

    while True:
        chunk = await queue.get()
        yield chunk
        if chunk == "data: [DONE]\n\n":
            break
