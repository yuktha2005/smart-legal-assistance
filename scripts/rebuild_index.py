import os
from src.vector_store import VectorStore
from src.document_ingestion import DocumentIngestion

def main():
    print("Rebuilding FAISS Index...")
    vs = VectorStore(
        index_path="models/faiss_index.bin",
        metadata_path="models/metadata.pkl"
    )
    ingestion = DocumentIngestion(vs)
    ingestion.load_default_documents()
    print("FAISS Index rebuilt successfully.")

if __name__ == "__main__":
    main()
