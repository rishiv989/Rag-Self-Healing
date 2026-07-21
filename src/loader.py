import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_documents(file_path):
    """
    Load documents from PDF, TXT, MD, or DOCX files with automatic fallback.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".txt", ".md", ".json", ".csv", ".log"]:
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            return loader.load()
        except Exception:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return [Document(page_content=text, metadata={"source": file_path})]

    # PDF loading with pdfplumber fallback
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if docs:
            return docs
    except Exception as e:
        print(f"[loader] PyPDFLoader error ({e}), attempting pdfplumber fallback...")

    # Fallback using pdfplumber
    try:
        import pdfplumber
        docs = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"source": file_path, "page": i}))
        if docs:
            return docs
    except Exception as e2:
        print(f"[loader] pdfplumber fallback error: {e2}")

    raise ValueError(f"Could not extract text from file '{os.path.basename(file_path)}'. Please ensure it is a valid PDF or TXT file.")


if __name__ == "__main__":
    pdf_path = "data/sample.pdf"

    docs = load_documents(pdf_path)

    print(f"Total pages loaded: {len(docs)}")
    print("\nFirst page preview:\n")
    print(docs[0].page_content[:500])
    print("\nMetadata:")
    print(docs[0].metadata)