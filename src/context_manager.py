"""Smart Legal Assistance - Phase 2 Module: Context Manager

Maintains conversational slots extracted from memory and retrieval results.

No external APIs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


SECTION_RE = re.compile(r"section\s*([0-9A-Za-z\-]+)", re.IGNORECASE)


@dataclass(frozen=True)
class ContextSlots:
    current_section: Optional[str] = None
    current_topic: Optional[str] = None
    current_document: Optional[str] = None


class ContextManager:
    def __init__(self) -> None:
        pass

    def extract_slots_from_history(self, memory_history: List[Dict[str, str]]) -> ContextSlots:
        # best-effort: find the most recent explicit section mention
        last_section: Optional[str] = None
        last_doc: Optional[str] = None

        for turn in reversed(memory_history or []):
            user = (turn.get("user") or "").strip()
            if not user:
                continue
            m = SECTION_RE.search(user)
            if m:
                last_section = m.group(1).upper()
                break

        # Document is typically in sources; here we can't see sources.
        return ContextSlots(current_section=last_section, current_document=last_doc)

    def build_retrieval_context(self, slots: ContextSlots) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if slots.current_section:
            out["current_section"] = slots.current_section
        if slots.current_document:
            out["current_document"] = slots.current_document
        return out

