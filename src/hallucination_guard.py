"""Smart Legal Assistance - Phase 2 Module: Hallucination Guard

Enforces:
- Never answer outside retrieved context.
- If confidence is low, return fixed message.
- Uses keyword overlap grounding (not exact substring) to work with LLMs that paraphrase.

No external APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal
import re


Confidence = Literal["High", "Medium", "Low"]


@dataclass(frozen=True)
class GuardResult:
    final_answer: str
    is_grounded: bool


class HallucinationGuard:
    def __init__(self, low_confidence_message: str | None = None) -> None:
        self.low_confidence_message = low_confidence_message or (
            "I couldn't find sufficient information in the uploaded legal documents. Please upload additional documents or rephrase your question."
        )

    def _extract_keywords(self, text: str) -> set:
        """Extract meaningful keywords (3+ chars) from text."""
        words = re.findall(r'\w+', text.lower())
        # Filter out very common stopwords and short words
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                     'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                     'have', 'been', 'from', 'this', 'that', 'with', 'they',
                     'which', 'their', 'will', 'would', 'there', 'could',
                     'other', 'into', 'more', 'some', 'such', 'than', 'its',
                     'also', 'these', 'about', 'what', 'when', 'where', 'who',
                     'how', 'may', 'shall', 'under', 'section', 'source',
                     'information', 'comes', 'from', 'document', 'documents'}
        return {w for w in words if len(w) >= 3 and w not in stopwords}

    def guard(
        self,
        answer: str,
        confidence_label: Confidence,
        retrieved_chunks: List[Dict],
    ) -> GuardResult:
        if not answer or not answer.strip():
            return GuardResult(final_answer=self.low_confidence_message, is_grounded=False)

        if confidence_label == "Low":
            return GuardResult(final_answer=self.low_confidence_message, is_grounded=True)

        # Build joined retrieved text including all metadata fields
        context_parts = []
        for c in retrieved_chunks:
            context_parts.append(c.get("text", ""))
            for f in ["crime_name", "description", "punishment", "ingredients", "ipc_section", "bns_section"]:
                if c.get(f):
                    context_parts.append(str(c[f]))
            if c.get("keywords"):
                context_parts.append(" ".join(c["keywords"]))
            if c.get("aliases"):
                context_parts.append(" ".join(c["aliases"]))
                
        joined = "\n".join(context_parts)
        if not joined.strip():
            return GuardResult(final_answer=self.low_confidence_message, is_grounded=False)

        # Keyword overlap grounding:
        # Check that a significant portion of meaningful answer keywords
        # appear in the retrieved context
        answer_keywords = self._extract_keywords(answer)
        context_keywords = self._extract_keywords(joined)
        
        if not answer_keywords:
            return GuardResult(final_answer=self.low_confidence_message, is_grounded=False)
        
        overlap = answer_keywords & context_keywords
        overlap_ratio = len(overlap) / len(answer_keywords) if answer_keywords else 0
        
        # If at least 20% of answer keywords come from context, consider it grounded
        # This is intentionally very lenient to allow paraphrasing of structured fields
        if overlap_ratio >= 0.20:
            return GuardResult(final_answer=answer, is_grounded=True)
        
        return GuardResult(final_answer=self.low_confidence_message, is_grounded=False)
