import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_ingestion import DocumentIngestion
from src.enhanced_rag_engine import EnhancedRAGEngine
from src.retriever import Retriever
from src.vector_store import VectorStore
from src.embedding_generator import EmbeddingGenerator
from src.enhanced_chatbot import EnhancedChatbot

def rebuild_index():
    print("Rebuilding FAISS index from data/documents...")
    
    # Initialize components
    embedder = EmbeddingGenerator()
    vector_store = VectorStore(
        index_path="models/faiss_index.bin",
        metadata_path="models/metadata.pkl",
        embedding_dim=384
    )
    
    ingestion = DocumentIngestion(vector_store)
    
    # Process all PDFs in legal_pdfs
    docs_dir = "legal_pdfs"
    total_chunks = 0
    
    for filename in os.listdir(docs_dir):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(docs_dir, filename)
            print(f"Processing {filename}...")
            chunks = ingestion.process_pdf(filepath)
            print(f"-> Generated {chunks} chunks")
            total_chunks += chunks
            
    print(f"\nTotal chunks in index: {total_chunks}")
    
    # Validate index
    vector_store.validate_index()
    return vector_store, embedder

def test_queries(vector_store, embedder):
    print("\n=== Running Acceptance Tests ===")
    
    retriever = Retriever(vector_store, embedder)
    rag_engine = EnhancedRAGEngine(retriever, model_name="qwen2.5:7b")
    chatbot = EnhancedChatbot(rag_engine)
    
    test_queries = [
        "What is the punishment for cheating?",
        "Difference between culpable homicide and murder",
        "What is the BNS section for Rape?"
    ]
    
    for q in test_queries:
        print(f"\nQ: {q}")
        response = chatbot.chat(q)
        try:
            print(f"A: {response['answer']}")
        except UnicodeEncodeError:
            print(f"A: {response['answer'].encode('ascii', 'replace').decode('ascii')}")
        print(f"Confidence: {response['confidence']:.2f}")
        print(f"Sources: {', '.join(response['sources'])}")
        print("-" * 50)

if __name__ == "__main__":
    vs, embedder = rebuild_index()
    test_queries(vs, embedder)
