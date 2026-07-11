import json
import os

from config.config import EMBEDDINGS_PATH, PROCESSED_CHUNKS_PATH, FAISS_INDEX_PATH, METADATA_PATH, TOP_K
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.rag_engine import RAGEngine


def test_rag_grounded_answer_has_sources():
    vs = VectorStore(index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH)
    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH)):
        # if index absent, fail fast (tests expect Phase-1 artifacts)
        raise AssertionError("FAISS index/metadata missing; run run_assistant.py --build-index")

    vs.load_index()

    from src.embedding_generator import EmbeddingGenerator
    embedder = EmbeddingGenerator()
    retriever = Retriever(vs, embedder)
    rag = RAGEngine(retriever=retriever, top_k=TOP_K)

    q = "What is the punishment for committing gang rape on a child under sixteen?"
    out = rag.answer_question(q)

    assert "answer" in out
    assert isinstance(out["answer"], str)
    assert len(out.get("sources", [])) >= 1
    assert "retrieved_chunks" in out

    # Groundedness check (answer must contain at least one sentence from retrieved chunks)
    retrieved = rag.retrieve_context(q)
    joined = "\n".join([c.get("text", "") for c in retrieved])
    # any non-empty answer sentence should appear
    import re
    sents = re.split(r"(?<=[\.!?])\s+", out["answer"].strip())
    sents = [s.strip() for s in sents if s.strip()]
    assert any(s in joined for s in sents)

