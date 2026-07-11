"""Smart Legal Assistance - Phase 2 Module 4

Legal Language Simplifier
---------------------------
Simplifies legal language into clearer plain English while preserving meaning.

Implemented as deterministic dictionary + safe term replacement rules.
No external APIs.
"""

from __future__ import annotations

import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class LegalSimplifier:
    def __init__(self):
        # Beginner-friendly dictionary mapping.
        # Important: keep section numbers/citations intact by only replacing
        # these standalone terms.
        self.dictionary: Dict[str, str] = {
            "petitioner": "the person filing the case",
            "respondent": "the other party in the case",
            "plaintiff": "the person who started the case",
            "defendant": "the person being sued",
            "injunction": "a court order to stop something",
            "mala fide": "in bad faith",
            "inter alia": "among other things",
            "ipso facto": "by the fact itself",
            "easementary rights": "rights connected to using another person's land",
            "tribunal": "the court or panel that hears the case",
            "breach": "breaking a duty or agreement",
        }

        # normalize keys for case-insensitive matching
        self._patterns = []
        for term, replacement in self.dictionary.items():
            # word-boundary safe for multi-word phrases too
            escaped = re.escape(term)
            pattern = re.compile(rf"\b{escaped}\b", flags=re.IGNORECASE)
            self._patterns.append((pattern, replacement))

    def replace_legal_terms(self, text: str) -> str:
        """Replace known legal terms in a case-insensitive way."""
        if not text:
            return text
        out = text
        for pattern, replacement in self._patterns:
            out = pattern.sub(replacement, out)
        return out

    def simplify(self, text: str) -> str:
        """Simplify legal paragraph into clearer plain English."""
        if not text or not text.strip():
            return ""

        simplified = self.replace_legal_terms(text)

        # Light cleanup: make sentences a bit easier to read.
        # Preserve meaning: avoid removing section numbers.
        simplified = re.sub(r"\s+", " ", simplified).strip()

        # If it still looks very formal, add small clarifying replacements.
        # (Keep deterministic and minimal.)
        simplified = re.sub(r"\bwhereas\b", "because", simplified, flags=re.IGNORECASE)
        simplified = re.sub(r"\bhereby\b", "now", simplified, flags=re.IGNORECASE)

        return simplified

