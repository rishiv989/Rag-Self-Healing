from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.splitter import split_documents


def create_vector_db(pdf_path):
    chunks = split_documents(pdf_path)

    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return vectorstore


if __name__ == "__main__":
    pdf_path = "data/sample.pdf"

    vectorstore = create_vector_db(pdf_path)

    print("Vector database created successfully!")
    print(f"Stored {vectorstore._collection.count()} chunks.")