import os
import re

from config.config import FAISS_INDEX_PATH, METADATA_PATH, TOP_K
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.rag_engine import RAGEngine


def test_rag_answer_is_sentence_grounded():
    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH)):
        raise AssertionError("FAISS index/metadata missing; run run_assistant.py --build-index")

    vs = VectorStore(index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH)
    vs.load_index()
    retriever = Retriever(vs)
    rag = RAGEngine(retriever=retriever, top_k=TOP_K)

    q = "What is section 376AB?"
    out = rag.answer_question(q)
    assert out["answer"]

    retrieved = rag.retrieve_context(q)
    joined = "\n".join([c.get("text", "") for c in retrieved])
    sents = [s.strip() for s in re.split(r"(?<=[\.!?])\s+", out["answer"].strip()) if s.strip()]
    assert any(s in joined for s in sents)

