"""Smart Legal Assistance - Phase 2 Module: Response Templates

Centralizes response strings for consistent UX.
"""

from __future__ import annotations


def greeting_template() -> str:
    return "Hello! I am your Smart Legal Assistant.\nHow can I help you today?"


def low_confidence_template() -> str:
    return "I could not find sufficient information in the legal documents to answer confidently."


def unknown_template() -> str:
    return "I’m not sure how to help with that request. Please ask a legal question, or say 'summarize' or 'simplify'."

