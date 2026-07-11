"""Command-line runner for Smart Legal Assistance (Phase 2 Backend Only).

Demonstrates:
- Build/load FAISS vector store
- Semantic retrieval
- Deterministic grounded RAG
- Summarization + simplification
- Conversation memory in multi-turn mode
- Evaluation report generation

No UI / no DB / no deployment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from typing import Any, Dict, List

import numpy as np

from src.vector_store import VectorStore
from src.retriever import Retriever
from src.rag_engine import RAGEngine
from src.summarizer import LegalSummarizer
from src.simplifier import LegalSimplifier
from src.memory import ConversationMemory
from src.enhanced_chatbot import EnhancedChatbot
from src.chatbot import LegalAssistant
from src.evaluation import EvaluationReport
from src.enhanced_chatbot import EnhancedChatbot

from config.config import (
    CONFIG,
    EMBEDDINGS_PATH,
    PROCESSED_CHUNKS_PATH,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    TOP_K,
    REPORT_PATH,
    LOG_PATH,
)


def setup_logging() -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )


def _load_phase1_metadata() -> List[Dict[str, Any]]:
    """Load processed_chunks.json and convert to required metadata schema."""
    with open(PROCESSED_CHUNKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    # processed_chunks.json stores: index/text/length/extra_metadata (includes chunk_id etc)
    # We convert into required metadata items.
    metadata: List[Dict[str, Any]] = []
    for ch in chunks:
        extra = ch.get("extra_metadata") or {}
        metadata.append(
            {
                "chunk_id": extra.get("chunk_id") if extra else f"CHUNK_{ch.get('index')}",
                "source_document": extra.get("doc_source") if extra else None,
                "text": ch.get("text"),
                "cluster": extra.get("cluster"),
                "topic": extra.get("topic"),
            }
        )

    # cluster/topic may be missing in extra_metadata; ensure keys exist.
    for m in metadata:
        m["cluster"] = m.get("cluster") if m.get("cluster") is not None else -1
        m["topic"] = m.get("topic") if m.get("topic") is not None else -1
        if m.get("source_document") is None:
            m["source_document"] = "unknown"

    return metadata


def build_or_load_vector_store(force_rebuild: bool = False) -> VectorStore:
    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = _load_phase1_metadata()

    vs = VectorStore(index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH, embedding_dim=embeddings.shape[1])

    if force_rebuild or not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH)):
        vs.build_index(embeddings=embeddings, metadata=metadata)
        vs.save_index()
    else:
        vs.load_index()

    return vs


def demo_retrieval(assistant, question: str, top_k: int = TOP_K) -> None:
    print("\n=== Retrieval Demo ===")
    result = assistant.ask(question)
    print("Q:", question)
    print("Answer:", result["answer"])
    print("Sources:")
    for s in assistant.show_sources():
        print(" -", s)


def demo_summarize_simplify(assistant, text: str) -> None:
    print("\n=== Summarize Demo ===")
    summary = assistant.summarize(text)
    print(summary)

    print("\n=== Simplify Demo ===")
    simplified = assistant.simplify(text)
    print(simplified)


def run_evaluation(vs: VectorStore) -> None:
    retriever = Retriever(vs)
    rag = RAGEngine(retriever=retriever, top_k=TOP_K)
    summarizer = LegalSummarizer()

    # Small evaluation set (groundedness checks extraction-only)
    # We don't have gold labels; expected_chunk_ids optional.
    questions = [
        "What is the punishment for gang rape on a child under sixteen under Section 376?",
        "What is section 376AB?",
    ]

    results: List[Dict[str, Any]] = []
    for q in questions:
        retrieved_chunks = rag.retrieve_context(q)
        # build sources_texts
        sources_texts = [c.get("text", "") for c in retrieved_chunks]
        out = rag.answer_question(q)

        # summarization compression (optional)
        original_text = retrieved_chunks[0].get("text", "") if retrieved_chunks else ""
        summary_text = summarizer.summarize_document(original_text) if original_text else ""

        results.append(
            {
                "question": q,
                "answer": out.get("answer", ""),
                "retrieved_chunks": out.get("retrieved_chunks", []),
                "sources_texts": sources_texts,
                "original_text": original_text,
                "summary_text": summary_text,
            }
        )

    er = EvaluationReport(
        output_md_path=REPORT_PATH + ".md",
        output_pdf_path=REPORT_PATH + ".pdf",
    )
    metrics = er.run_and_write(results)
    print("\nEvaluation complete. Metrics:")
    print(json.dumps(metrics, indent=2))


def run_chatbot_multi_turn(vs: VectorStore) -> None:
    retriever = Retriever(vs)
    rag = RAGEngine(retriever=retriever, top_k=TOP_K)

    memory_path = os.path.join("logs", "conversation_history.json")

    memory = ConversationMemory(memory_path=memory_path)

    assistant = EnhancedChatbot(
        rag_engine=rag,
        summarizer=LegalSummarizer(),
        simplifier=LegalSimplifier(),
        memory=memory,
    )





    print("\n=== Legal Assistant (Phase 2 Backend) ===")
    print("Type 'exit' to quit. Type 'clear' to clear history.")

    while True:
        user = input("You: ").strip()
        if user.lower() == "exit":
            break
        if user.lower() == "clear":
            assistant.clear_history()
            print("Assistant: history cleared.")
            continue

        result = assistant.ask(user)
        print("Assistant:")
        print(result.get("answer", ""))
        print("Sources:")
        for s in assistant.show_sources():
            print(" -", s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild FAISS index")
    parser.add_argument("--build-index", action="store_true", help="Force rebuild index")
    parser.add_argument("--evaluate", action="store_true", help="Run Phase 2 evaluation")
    parser.add_argument("--demo", action="store_true", help="Run fixed demos")
    parser.add_argument("--chat", action="store_true", help="Interactive multi-turn chatbot")
    args = parser.parse_args()

    setup_logging()

    vs = build_or_load_vector_store(force_rebuild=args.force_rebuild or args.build_index)

    if args.evaluate:
        run_evaluation(vs)
        return

    if args.demo:
        retriever = Retriever(vs)
        rag = RAGEngine(retriever=retriever, top_k=TOP_K)
        assistant = LegalAssistant(
            rag_engine=rag,
            summarizer=LegalSummarizer(),
            simplifier=LegalSimplifier(),
            memory=ConversationMemory(memory_path=os.path.join("logs", "conversation_history.json")),
        )

        demo_retrieval(
            assistant,
            "What is the punishment for committing gang rape on a child under sixteen?",
            top_k=TOP_K,
        )

        # Simplify demo
        sample_text = "The petitioner seeks permanent injunction." \
                       " The tribunal found a breach of the agreement." 
        demo_summarize_simplify(assistant, sample_text)
        return

    if args.chat:
        run_chatbot_multi_turn(vs)
        return

    # default: print quick usage
    print("Run with --demo, --chat, or --evaluate")


if __name__ == "__main__":
    main()

