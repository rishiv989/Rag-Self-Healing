import src.state as state

# Phrases that indicate the document-grounded answer couldn't find the info.
# If the draft answer contains these, we fall back to LLM general knowledge.
NOT_IN_DOCUMENT_PHRASES = [
    "not mentioned",
    "not explicitly mentioned",
    "not in the provided document",
    "not in the document",
    "does not mention",
    "doesn't mention",
    "not included in the document",
    "no mention of",
    "not discussed in",
    "not covered in",
    "not found in the document",
    "not referenced in",
]


def answer_not_in_document(draft_answer: str) -> bool:
    """Returns True if the draft answer says the topic isn't in the documents."""
    lower = draft_answer.lower()
    return any(phrase in lower for phrase in NOT_IN_DOCUMENT_PHRASES)


async def generate_memory_answer_stream(query):
    memory_context = state.chat_history[-1]["assistant"] if state.chat_history else ""

    prompt = f"""
You are answering ONLY from prior grounded conversation memory.
Rules:
- answer directly
- concise
- no conversational filler
- do not ask follow-up questions
- do not invent facts

Previous grounded assistant response:
{memory_context}

Current user question:
{query}
"""
    try:
        async for chunk in state.llm.astream(prompt):
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            yield text
        return
    except Exception as e:
        print(f"[generation] memory astream error: {e}")

    try:
        res = await state.llm.ainvoke(prompt)
        yield res.content if hasattr(res, "content") else str(res)
    except Exception as e:
        print(f"[generation] memory ainvoke error: {e}")
        yield f"LLM Generation Error: {str(e)}"


async def generate_answer_stream(context, query):
    prompt = f"""You are a document-grounded AI assistant.

Document Context:
{context}

Question:
{query}

Instructions:
Answer the question using the Document Context. Always cite your sources inline using bracketed numbers, e.g. [1] or [2].
Do not list the sources at the end, just use the inline citations.
"""
    try:
        async for chunk in state.llm.astream(prompt):
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            yield text
        return
    except Exception as e:
        print(f"[generation] generate_answer_stream astream error: {e}")

    try:
        res = await state.llm.ainvoke(prompt)
        yield res.content if hasattr(res, "content") else str(res)
    except Exception as e:
        print(f"[generation] generate_answer_stream ainvoke error: {e}")
        yield f"LLM Generation Error: {str(e)}"


async def correct_answer_stream(context, query, draft_answer):
    prompt = f"""
Question:
{query}

Document Context:
{context}

Previous Weak Answer:
{draft_answer}
"""
    try:
        async for chunk in state.llm.astream(prompt):
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            yield text
        return
    except Exception as e:
        print(f"[generation] correct_answer_stream astream error: {e}")

    try:
        res = await state.llm.ainvoke(prompt)
        yield res.content if hasattr(res, "content") else str(res)
    except Exception as e:
        print(f"[generation] correct_answer_stream ainvoke error: {e}")
        yield f"LLM Generation Error: {str(e)}"


async def generate_general_knowledge_stream(query):
    """
    Answer using LLM's general training knowledge when document context
    doesn't contain the answer. Called as fallback by rag_engine.py when
    the document-grounded draft answer says 'not mentioned'.
    """
    prompt = f"""You are a knowledgeable AI assistant. The user's uploaded documents do not contain information about this topic.

Answer the following question from your general training knowledge. Be accurate, clear, and helpful.

Question: {query}

Start your answer with: "*(Not found in your documents — answering from general knowledge)*\\n\\n"
"""
    try:
        async for chunk in state.llm.astream(prompt):
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            yield text
        return
    except Exception as e:
        print(f"[generation] general_knowledge astream error: {e}")

    try:
        res = await state.llm.ainvoke(prompt)
        yield res.content if hasattr(res, "content") else str(res)
    except Exception as e:
        print(f"[generation] general_knowledge ainvoke error: {e}")
        yield f"LLM Generation Error: {str(e)}"
