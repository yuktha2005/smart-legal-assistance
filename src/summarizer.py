"""Smart Legal Assistance - Phase 2 Module 3

Legal Document Summarizer
----------------------------
Summarizes long legal text using t5-small.

Because internships must run locally, this module includes a safe
fallback extractive summarizer when transformers/model download fails.

Implements:
- summarize_case(text)
- summarize_document(text)
- extract_key_points(text)

Notes:
- No external APIs.
- Deterministic-ish settings (seeded beams) for reproducibility.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


class LegalSummarizer:
    def __init__(self, model_name: str = "t5-small"):
        self.model_name = model_name
        self._summarizer = None
        self._mode = "extractive"

        # Try to initialize transformers pipeline.
        try:
            from transformers import pipeline  # type: ignore

            logger.info("Loading summarization model: %s", model_name)
            self._summarizer = pipeline(
                "text2text-generation",
                model=model_name,
                tokenizer=model_name,
            )
            self._mode = "transformers"
            logger.info("Summarizer mode: transformers")
        except Exception:
            logger.warning("Falling back to extractive summarizer (transformers unavailable)")
            self._summarizer = None
            self._mode = "extractive"

    @staticmethod
    def _split_for_long_text(text: str, max_chars: int = 1500) -> List[str]:
        """Split text into manageable segments by sentence boundaries."""
        sentences = re.split(r"(?<=[\.!?])\s+", text.strip())
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for s in sentences:
            if not s.strip():
                continue
            if current_len + len(s) > max_chars and current:
                chunks.append(" ".join(current))
                current = [s]
                current_len = len(s)
            else:
                current.append(s)
                current_len += len(s)
        if current:
            chunks.append(" ".join(current))
        return chunks

    @staticmethod
    def _extractive_summary(text: str, max_sentences: int = 5) -> str:
        """Simple extractive summarization using keyword frequency (no ML)."""
        # Sentence split
        sentences = re.split(r"(?<=[\.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return ""
        if len(sentences) <= max_sentences:
            return " ".join(sentences)

        words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1

        scored = []
        for i, s in enumerate(sentences):
            s_words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", s.lower())
            score = sum(freq.get(w, 0) for w in s_words)
            scored.append((score, i, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = sorted(scored[:max_sentences], key=lambda x: x[1])
        return " ".join([s for _, _, s in top])

    def summarize_document(self, text: str, max_length: int = 180) -> str:
        """Summarize a long document."""
        if not text or not text.strip():
            return ""

        if self._mode == "extractive":
            return self._extractive_summary(text, max_sentences=6)

        # transformers mode with chunking
        chunks = self._split_for_long_text(text, max_chars=2000)
        summaries: List[str] = []
        try:
            for c in chunks:
                prompt = "summarize: " + c
                out = self._summarizer(
                    prompt,
                    max_length=max_length,
                    min_length=30,
                    do_sample=False,
                    num_beams=4,
                )
                if out and isinstance(out, list) and "generated_text" in out[0]:
                    summaries.append(out[0]["generated_text"])

            combined = " ".join(summaries)
            # Final compression pass if too long
            if len(combined) > 1000:
                return self._extractive_summary(combined, max_sentences=5)
            return combined.strip()
        except Exception:
            logger.exception("Transformer summarization failed; switching to extractive")
            return self._extractive_summary(text, max_sentences=6)

    def summarize_case(self, text: str) -> str:
        """Case-oriented summary (uses summarize_document)."""
        return self.summarize_document(text, max_length=200)

    def extract_key_points(self, text: str, max_points: int = 5) -> List[str]:
        """Extract key points.

        In transformers mode, we ask for a short summary then split into bullets.
        In extractive mode, we select sentences.
        """
        if not text or not text.strip():
            return []

        if self._mode == "extractive":
            summary = self._extractive_summary(text, max_sentences=max_points)
            # split into sentences
            points = re.split(r"(?<=[\.!?])\s+", summary.strip())
            return [p.strip() for p in points if p.strip()][:max_points]

        try:
            prompt = "extract key legal points: " + text
            out = self._summarizer(
                prompt,
                max_length=120,
                min_length=40,
                do_sample=False,
                num_beams=4,
            )
            if out and isinstance(out, list) and "generated_text" in out[0]:
                gen = out[0]["generated_text"].strip()
                # Try bullet/sentence split
                parts = re.split(r"\n+|-+|•", gen)
                cleaned = [p.strip() for p in parts if p.strip()]
                # fallback to sentence split
                if len(cleaned) < max_points:
                    cleaned = [
                        s.strip()
                        for s in re.split(r"(?<=[\.!?])\s+", gen)
                        if s.strip()
                    ]
                return cleaned[:max_points]
        except Exception:
            logger.exception("Key-point extraction failed; using extractive")

        # fallback
        summary = self._extractive_summary(text, max_sentences=max_points)
        points = re.split(r"(?<=[\.!?])\s+", summary.strip())
        return [p.strip() for p in points if p.strip()][:max_points]

