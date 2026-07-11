
"""Smart Legal Assistance - Phase 2 Module 5

Retrieval Augmented Generation (RAG) - grounded & no-hallucination mode
-------------------------------------------------------------------------
No external LLMs.

The answer is generated ONLY from retrieved chunk texts.
We implement a deterministic "extractive" RAG:
- retrieve_context(): pack top-k retrieved chunks
- generate_answer(): pick sentences from retrieved chunks that best match question
- answer_question(): returns answer + citations (source chunks)

Output schema:
{
    "answer": "...",
    "sources": [],
    "similarities": [],
    "retrieved_chunks": []
}

Groundedness guarantee:
- Every sentence in `answer` must come verbatim from at least one retrieved chunk.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from src.retriever import Retriever


logger = logging.getLogger(__name__)


def _sentence_split(text: str) -> List[str]:
    """Split into sentence-like units.

    The unit tests split the answer using this exact regex:
        r"(?<=[\.!?])\s+"

    We use the same rule here to keep grounding checks consistent.
    """
    parts = re.split(r"(?<=[\.!?])\\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]



class RAGEngine:
    def __init__(self, retriever: Retriever, top_k: int = 5):
        self.retriever = retriever
        self.top_k = top_k

        # Keep deterministic behavior
        self.max_answer_sentences = 5

        # Simple relevance metric
        self._token_re = re.compile(r"\b[a-zA-Z0-9_/-]{3,}\b")

    def retrieve_context(self, question: str) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks for a question."""
        return self.retriever.retrieve_top_k(question, top_k=self.top_k)

    def _score_sentence(self, question_tokens: set[str], sentence: str) -> int:
        tokens = set(self._token_re.findall(sentence.lower()))
        return len(tokens & question_tokens)

    def generate_answer(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """Generate a grounded answer by extracting best-matching sentences."""
        if not retrieved_chunks:
            return "I could not find relevant information in the available documents.", []

        q_tokens = set(self._token_re.findall(question.lower()))
        if not q_tokens:
            return "I could not understand the question terms to search the documents.", []

        candidates: List[Tuple[int, int, str, Dict[str, Any]]] = []
        # (score, chunk_idx, sentence, chunk_meta)
        for chunk_idx, ch in enumerate(retrieved_chunks):
            chunk_text = ch.get("text", "")
            for sent in _sentence_split(chunk_text):
                score = self._score_sentence(q_tokens, sent)
                if score > 0:
                    candidates.append((score, chunk_idx, sent, ch))

        if not candidates:
            # Grounded fallback: return the top chunk text as-is.
            first_chunk = retrieved_chunks[0]
            cid = first_chunk.get("chunk_id", "")
            return (first_chunk.get("text", "") or ""), [cid]

        # Sort deterministically: higher score first, then earlier chunk index, then sentence length
        candidates.sort(key=lambda x: (-x[0], x[1], -len(x[2])))
        chosen = candidates[: self.max_answer_sentences]


        # Build answer directly from chosen sentences.
        # Tests split answer into sentences using punctuation regex and then
        # checks whether each split sentence is a substring of the joined
        # retrieved chunk texts. To satisfy this strictly, we must ensure
        # the answer contains sentences that appear verbatim in the retrieved
        # chunk texts (including punctuation).
        answer_sents: List[str] = []
        seen = set()
        for _, _, sent, _ch in chosen:
            if sent in seen:
                continue
            seen.add(sent)
            answer_sents.append(sent)

        answer = " ".join(answer_sents).strip()



        source_chunk_ids = []
        for _score, _idx, _sent, ch in chosen:
            cid = ch.get("chunk_id")
            if cid is not None and cid not in source_chunk_ids:
                source_chunk_ids.append(cid)

        return answer, source_chunk_ids

    def answer_question(self, question: str) -> Dict[str, Any]:
        retrieved_chunks = self.retrieve_context(question)

        answer, source_chunk_ids = self.generate_answer(question, retrieved_chunks)

        sources = [
            {
                "chunk_id": ch.get("chunk_id"),
                "source_document": ch.get("source_document"),
                "cluster": ch.get("cluster"),
                "topic": ch.get("topic"),
            }
            for ch in retrieved_chunks
            if ch.get("chunk_id") in source_chunk_ids
        ]

        return {
            "answer": answer,
            "sources": sources,
            "similarities": [ch.get("similarity") for ch in retrieved_chunks],
            "retrieved_chunks": [
                {
                    "chunk_id": ch.get("chunk_id"),
                    "source_document": ch.get("source_document"),
                    "similarity": ch.get("similarity"),
                }
                for ch in retrieved_chunks
            ],
        }

