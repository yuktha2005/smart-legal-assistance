"""Smart Legal Assistance - Phase 2 Module 6

Conversation Memory
--------------------
Stores conversation history in JSON format on local disk.

History entry schema:
{
  "user": "...",
  "assistant": "...",
  "timestamp": "ISO-8601"
}

Also includes a small helper to produce a context string.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging


logger = logging.getLogger(__name__)


class ConversationMemory:
    def __init__(self, memory_path: str):
        self.memory_path = memory_path
        self.history: List[Dict[str, Any]] = []
        self.state: Dict[str, Any] = {}

    def load(self) -> None:
        """Load history and state from disk (if exists)."""
        try:
            if not os.path.exists(self.memory_path):
                self.history = []
                self.state = {}
                return
            with open(self.memory_path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
                # Handle old schema (just list) vs new schema (dict with history and state)
                if isinstance(data, list):
                    self.history = data
                    self.state = {}
                else:
                    self.history = data.get("history", [])
                    self.state = data.get("state", {})
            logger.info("Loaded conversation memory (%d turns)", len(self.history))
        except Exception:
            logger.exception("Failed to load conversation memory")
            raise

    def save(self) -> None:
        """Save history and state to disk."""
        try:
            os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
            with open(self.memory_path, "w", encoding="utf-8") as f:
                data = {
                    "history": self.history,
                    "state": self.state
                }
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info("Saved conversation memory (%d turns)", len(self.history))
        except Exception:
            logger.exception("Failed to save conversation memory")
            raise

    def clear(self) -> None:
        """Clear in-memory and persisted history."""
        self.history = []
        self.state = {}
        if os.path.exists(self.memory_path):
            try:
                os.remove(self.memory_path)
            except OSError:
                # don't fail hard
                logger.warning("Could not delete memory file at %s", self.memory_path)

    def add_turn(self, user: str, assistant: str) -> None:
        self.history.append(
            {
                "user": user,
                "assistant": assistant,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def to_context_string(self, max_turns: int = 6) -> str:
        """Convert memory into a compact context string."""
        if not self.history:
            return ""
        turns = self.history[-max_turns:]
        lines = []
        for t in turns:
            lines.append(f"User: {t.get('user','')}")
            lines.append(f"Assistant: {t.get('assistant','')}")
        return "\n".join(lines).strip()

