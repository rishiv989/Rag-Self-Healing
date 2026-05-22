import asyncio
from langchain_core.documents import Document
import src.state as state

def filter_docs(docs):
    filtered = []
    for doc in docs:
        text = doc.page_content.lower()
        if "table of contents" in text:
            continue
        if len(text.strip()) < 100:
            continue
        filtered.append(doc)
    return filtered


async def vector_retrieve(search_query, search_type="similarity"):
    if search_type == "mmr":
        retriever = state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.5}
        )
    else:
        retriever = state.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )

    # Use ainvoke for async execution
    docs = await retriever.ainvoke(search_query)
    return filter_docs(docs)


async def keyword_retrieve(search_query):
    if state.bm25 is None:
        return []

    # Run CPU-bound bm25 synchronously in a thread or just normally (it's very fast)
    def do_search():
        tokenized_query = search_query.lower().split()
        scores = state.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        keyword_docs = []
        for idx in ranked_indices:
            keyword_docs.append(
                Document(page_content=state.ALL_DOCS[idx], metadata=state.ALL_METADATA[idx])
            )
        return filter_docs(keyword_docs)
    
    return await asyncio.to_thread(do_search)


async def retrieve_documents(search_query, search_type="hybrid"):
    if search_type == "hybrid":
        # Run both retrievals concurrently
        vector_task = asyncio.create_task(vector_retrieve(search_query))
        keyword_task = asyncio.create_task(keyword_retrieve(search_query))
        
        vector_docs, keyword_docs = await asyncio.gather(vector_task, keyword_task)

        combined = vector_docs + keyword_docs
        unique = {}
        for doc in combined:
            unique[doc.page_content] = doc

        return list(unique.values())

    elif search_type == "mmr":
        return await vector_retrieve(search_query, "mmr")

    return await vector_retrieve(search_query)
