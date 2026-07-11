import os

from config.config import FAISS_INDEX_PATH, METADATA_PATH, TOP_K
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.rag_engine import RAGEngine
from src.enhanced_chatbot import EnhancedChatbot
from src.summarizer import LegalSummarizer
from src.simplifier import LegalSimplifier
from src.memory import ConversationMemory


def test_enhanced_chatbot_greeting_bypasses_retrieval(tmp_path):
    # Ensure memory works; retrieval artifacts must exist for other tests, but greeting should not need them.
    mem_path = tmp_path / "mem.json"

    # minimal components for chatbot init
    # For greeting bypass, rag_engine isn't called; but we still need to build it.
    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH)):
        raise AssertionError("FAISS index/metadata missing; run run_assistant.py --build-index")

    vs = VectorStore(index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH)
    vs.load_index()
    retriever = Retriever(vs)
    rag = RAGEngine(retriever=retriever, top_k=TOP_K)

    assistant = EnhancedChatbot(
        rag_engine=rag,
        summarizer=LegalSummarizer(),
        simplifier=LegalSimplifier(),
        memory=ConversationMemory(memory_path=str(mem_path)),
    )

    out = assistant.ask("hi")
    assert "Smart Legal Assistant" in out["answer"]
    assert out.get("sources") == []

