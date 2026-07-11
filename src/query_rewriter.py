import re
from dataclasses import dataclass
from src.legal_term_mapper import LegalTermMapper
from src.intent_classifier import IntentClassifier, IntentResult

@dataclass(frozen=True)
class RewriteResult:
    rewritten_query: str
    intent: str = "general_query"

class QueryRewriter:
    def __init__(self):
        self.term_mapper = LegalTermMapper()
        self.intent_classifier = IntentClassifier()

    def rewrite(self, user_message: str, context: dict) -> RewriteResult:
        text = (user_message or "").strip()
        if not text:
            return RewriteResult(rewritten_query="", intent="general_query")

        q_lower = text.lower()
        intent_obj = self.intent_classifier.get_intent(text)
        intent = str(intent_obj)
        
        current_topic = context.get("current_topic")
        current_section = context.get("current_section")
        
        # Follow-up pronoun resolution
        has_pronoun = any(word in q_lower.split() for word in ['it', 'its', 'this', 'that', 'they', 'he', 'she'])
        
        rewritten = text
        if has_pronoun:
            if current_topic:
                if re.fullmatch(r".*\b(its|it|that)\b.*\bpunishment|.*\bpenalty\b", text, re.IGNORECASE):
                    rewritten = f"What is the punishment for {current_topic}?"
                else:
                    rewritten = f"Regarding {current_topic}: {text}"
            elif current_section:
                if re.fullmatch(r".*\b(its|it|that)\b.*\bpunishment|.*\bpenalty\b", text, re.IGNORECASE):
                    rewritten = f"What is the punishment under section {current_section}?"
                else:
                    rewritten = f"Regarding section {current_section}: {text}"

        # Scenario query rewriting:
        SCENARIO_MAPPINGS = {
            r"\b(forg|cheque|valuable security|will)\b": ["Forgery", "IPC Section 467", "IPC Section 463", "IPC Section 465", "BNS Section 336", "BNS Section 338"],
            r"\b(dishonestly keeps|keeps another|misappropriate|misappropriat|own use)\b": ["Dishonest Misappropriation of Property", "IPC Section 403", "BNS Section 314"],
            r"\b(threaten|fear of injury|fear of hurt|puts them in fear)\b": ["Extortion", "IPC Section 383", "IPC Section 384", "BNS Section 308"],
            r"\b(breach of trust|entrusted|entrustment)\b": ["Criminal Breach of Trust", "IPC Section 405", "IPC Section 406", "BNS Section 316"],
            r"\b(culpable homicide|homicide|causing death)\b": ["Culpable Homicide", "IPC Section 299", "IPC Section 300", "BNS Section 101"],
            r"\b(gang rape|rape|minor gang rape)\b": ["Rape", "Gang Rape", "IPC Section 376", "IPC Section 376DA", "BNS Section 64", "BNS Section 70(2)"],
            r"\b(theft|steal|stolen|movable property without consent)\b": ["Theft", "IPC Section 303", "BNS Section 303"],
        }

        # Check if the intent is classified as Scenario, or check keywords in the text
        is_scenario = (intent_obj.sub_intent == "Scenario Query" or "forged" in q_lower or "dishonestly keeps" in q_lower or "threatens" in q_lower)
        
        if is_scenario:
            mapped_concepts = []
            for pattern, concepts in SCENARIO_MAPPINGS.items():
                if re.search(pattern, q_lower):
                    mapped_concepts.extend(concepts)
            if mapped_concepts:
                concepts_str = ", ".join(mapped_concepts)
                # Append mapped legal concepts to the query to guide retrieval
                rewritten = f"{rewritten} (Legal Concepts: {concepts_str})"

        # Map terms to sections for better retrieval
        rewritten = self.term_mapper.map_terms(rewritten)
        
        return RewriteResult(rewritten_query=rewritten, intent=intent)
