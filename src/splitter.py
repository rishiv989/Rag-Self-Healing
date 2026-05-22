from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from src.loader import load_documents

def split_documents(pdf_path):
    docs = load_documents(pdf_path)

    # Use the same embeddings model used for VectorDB
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    # Semantic chunking relies on vector distance. 
    # The 'percentile' threshold forces a split when semantic distance between sentences is high.
    splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=80
    )

    chunks = splitter.split_documents(docs)

    return chunks


if __name__ == "__main__":
    pdf_path = "data/sample.pdf"

    chunks = split_documents(pdf_path)

    print("Original pages:", len(load_documents(pdf_path)))
    print("Total chunks created:", len(chunks))

    print("\nFirst chunk:\n")
    print(chunks[0].page_content)