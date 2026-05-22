from langchain_community.document_loaders import PyPDFLoader


def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


if __name__ == "__main__":
    pdf_path = "data/sample.pdf"

    docs = load_documents(pdf_path)

    print(f"Total pages loaded: {len(docs)}")
    print("\nFirst page preview:\n")
    print(docs[0].page_content[:500])
    print("\nMetadata:")
    print(docs[0].metadata)