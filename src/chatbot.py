"""Smart Legal Assistance - Phase 2 Module 7

Legal Chatbot Backend
----------------------
Pure backend logic for multi-turn legal Q&A.

Supports commands via methods:
- ask(user_message)
- summarize(text)
- simplify(text)
- show_sources()
- clear_history()

Conversation history is preserved through ConversationMemory.

No UI / no DB.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from src.rag_engine import RAGEngine
from src.summarizer import LegalSummarizer
from src.simplifier import LegalSimplifier
from src.memory import ConversationMemory


logger = logging.getLogger(__name__)


class LegalAssistant:
    def __init__(
        self,
        rag_engine: RAGEngine,
        summarizer: Optional[LegalSummarizer] = None,
        simplifier: Optional[LegalSimplifier] = None,
        memory: Optional[ConversationMemory] = None,
    ):
        self.rag_engine = rag_engine
        self.summarizer = summarizer or LegalSummarizer()
        self.simplifier = simplifier or LegalSimplifier()
        self.memory = memory or ConversationMemory(memory_path=os.path.join("logs", "conversation_history.json"))
        self.memory.load()

        self.last_sources: List[Dict[str, Any]] = []
        self.last_answer: str = ""

    def ask(self, user_message: str) -> Dict[str, Any]:
        """Main entry: answer a user question grounded in retrieved chunks."""
        if not user_message or not user_message.strip():
            return {"answer": "Please provide a question.", "sources": [], "retrieved_chunks": []}

        # Context-aware query: include prior history to help retrieval.
        context = self.memory.to_context_string()
        query = user_message
        if context:
            # Keep short; retrieval embedding will still work.
            query = context + "\n\nCurrent question: " + user_message

        result = self.rag_engine.answer_question(query)
        answer = result.get("answer", "")

        self.last_sources = result.get("sources", [])
        self.last_answer = answer

        # Save conversation turn
        self.memory.add_turn(user=user_message, assistant=answer)
        self.memory.save()

        return result

    def summarize(self, text: str) -> str:
        return self.summarizer.summarize_document(text)

    def simplify(self, text: str) -> str:
        return self.simplifier.simplify(text)

    def show_sources(self) -> List[Dict[str, Any]]:
        return self.last_sources

    def clear_history(self) -> None:
        self.memory.clear()
        self.last_sources = []
        self.last_answer = ""

