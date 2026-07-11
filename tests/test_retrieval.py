import os
import json
import numpy as np

from config.config import EMBEDDINGS_PATH, PROCESSED_CHUNKS_PATH, FAISS_INDEX_PATH, METADATA_PATH, TOP_K
from src.vector_store import VectorStore
from src.retriever import Retriever


def _load_metadata():
    with open(PROCESSED_CHUNKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data.get("chunks", [])
    out = []
    for ch in chunks:
        extra = ch.get("extra_metadata") or {}
        out.append(
            {
                "chunk_id": extra.get("chunk_id") if extra else f"CHUNK_{ch.get('index')}",
                "source_document": extra.get("doc_source") if extra else "unknown",
                "text": ch.get("text"),
                "cluster": extra.get("cluster", -1) if extra else -1,
                "topic": extra.get("topic", -1) if extra else -1,
            }
        )
    for m in out:
        m["cluster"] = m.get("cluster", -1)
        m["topic"] = m.get("topic", -1)
        if m.get("source_document") is None:
            m["source_document"] = "unknown"
    return out


def test_retrieval_top_k_smoke():
    assert os.path.exists(EMBEDDINGS_PATH)
    assert os.path.exists(PROCESSED_CHUNKS_PATH)

    vs = VectorStore(index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH)

    # Build if missing
    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH)):
        emb = np.load(EMBEDDINGS_PATH)
        meta = _load_metadata()
        vs.build_index(emb, meta)
        vs.save_index()
    else:
        vs.load_index()

    retriever = Retriever(vs)
    results = retriever.retrieve_top_k("What is punishment under section 376AB?", top_k=3)

    assert isinstance(results, list)
    assert len(results) == 3
    for r in results:
        assert "chunk_id" in r
        assert "source_document" in r
        assert "text" in r or r.get("text") is None
        assert "similarity" in r

