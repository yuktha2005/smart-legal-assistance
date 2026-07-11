"""Smart Legal Assistance - Phase 2 Module: Confidence Estimation

Computes High/Medium/Low confidence based on:
- top similarity score
- number of supporting chunks
- agreement among chunks (shared keywords)

No external APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal
import re


Confidence = Literal["High", "Medium", "Low"]


@dataclass(frozen=True)
class ConfidenceResult:
    label: Confidence
    reason: str


class ConfidenceEstimator:
    def __init__(
        self,
        high_similarity: float = 0.75,
        medium_similarity: float = 0.55,
        min_supporting_chunks_for_high: int = 2,
    ) -> None:
        self.high_similarity = high_similarity
        self.medium_similarity = medium_similarity
        self.min_supporting_chunks_for_high = min_supporting_chunks_for_high
        self._token_re = re.compile(r"\b[a-zA-Z0-9_/-]{3,}\b")

    def estimate(self, retrieved_chunks: List[Dict]) -> ConfidenceResult:
        if not retrieved_chunks:
            return ConfidenceResult(label="Low", reason="No retrieved chunks")

        sims = [c.get("similarity") for c in retrieved_chunks if c.get("similarity") is not None]
        if not sims:
            return ConfidenceResult(label="Low", reason="No similarity scores")

        top_sim = float(max(sims))
        supporting = sum(1 for s in sims if float(s) >= self.medium_similarity)

        # Keyword agreement: overlap of top chunks' tokens
        tokens = []
        for c in retrieved_chunks[:3]:
            txt = (c.get("text") or "")
            tokens.append(set(self._token_re.findall(txt.lower())))
        if tokens:
            inter = set.intersection(*tokens) if len(tokens) > 1 else tokens[0]
            agreement = len(inter)
        else:
            agreement = 0

        if top_sim >= self.high_similarity and supporting >= self.min_supporting_chunks_for_high:
            return ConfidenceResult(label="High", reason=f"Top similarity={top_sim:.3f}, supporting={supporting}, agreement={agreement}")

        if top_sim >= self.medium_similarity and supporting >= 1:
            return ConfidenceResult(label="Medium", reason=f"Top similarity={top_sim:.3f}, supporting={supporting}, agreement={agreement}")

        return ConfidenceResult(label="Low", reason=f"Top similarity={top_sim:.3f}, supporting={supporting}, agreement={agreement}")

