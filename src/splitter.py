from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.loader import load_documents

def split_documents(pdf_path, embeddings=None):
    """
    Fast, reliable document chunker optimized for production web APIs.
    Uses RecursiveCharacterTextSplitter for instant (<0.1s) chunking.
    """
    docs = load_documents(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
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