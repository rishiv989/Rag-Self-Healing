import src.state as state


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
    prompt = f"""
You are a document-grounded AI assistant.

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
