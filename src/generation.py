import src.state as state

async def generate_memory_answer_stream(query):
    memory_context = state.chat_history[-1]["assistant"]

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
    async for chunk in state.llm.astream(prompt):
        yield chunk.content


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
    async for chunk in state.llm.astream(prompt):
        yield chunk.content


async def correct_answer_stream(context, query, draft_answer):
    prompt = f"""
Question:
{query}

Document Context:
{context}

Previous Weak Answer:
{draft_answer}
"""
    async for chunk in state.llm.astream(prompt):
        yield chunk.content
