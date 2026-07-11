import os
import logging
from src.enhanced_rag_engine import EnhancedRAGEngine
from src.query_rewriter import QueryRewriter
from src.memory_manager import MemoryManager
from src.memory import ConversationMemory

class EnhancedChatbot:
    def __init__(self, rag_engine: EnhancedRAGEngine, summarizer=None, simplifier=None, memory=None):
        self.rag_engine = rag_engine
        self.query_rewriter = QueryRewriter()
        
        # Initialize memory
        if memory is None:
            mem_path = os.path.join("logs", "conversation_history.json")
            base_memory = ConversationMemory(memory_path=mem_path)
            base_memory.load()
            self.memory_manager = MemoryManager(base_memory)
        else:
            self.memory_manager = MemoryManager(memory)

    def ask(self, user_query: str, session_history: list[dict] = None, last_sources: list = None) -> dict:
        return self.chat(user_query, session_history=session_history, last_sources=last_sources)

    def chat(self, user_query: str, session_history: list[dict] = None, last_sources: list = None) -> dict:
        """
        Main entry point for chat interactions.
        """
        try:
            q_lower = user_query.lower().strip()
            state = self.memory_manager.get_state()
            
            # Strict intent: Where did you get this information?
            source_phrases = ["where did you get", "what is the source", "source", "where is it from"]
            if any(phrase in q_lower for phrase in source_phrases):
                current_doc = state.get("current_document")
                if current_doc:
                    return {
                        "answer": f"This information comes from {current_doc}.",
                        "sources": [current_doc],
                        "confidence": 1.0
                    }
                elif last_sources:
                    sources_str = ", ".join(last_sources)
                    return {
                        "answer": f"This information comes from {sources_str}.",
                        "sources": last_sources,
                        "confidence": 1.0
                    }
                else:
                    return {
                        "answer": "I don't have a specific source document referenced for the previous answer.",
                        "sources": [],
                        "confidence": 1.0
                    }
                    
            # Rewrite query using state and get intent
            rewrite_result = self.query_rewriter.rewrite(user_query, state)
            rewritten_query = rewrite_result.rewritten_query
            intent = rewrite_result.intent
            
            # Early exit for greetings
            if intent.upper() == 'GREETING':
                return {
                    "answer": "Hello! I'm your Smart Legal Assistant.\nHow can I help you today?",
                    "sources": [],
                    "confidence": 1.0
                }
            
            # Get formatted history
            history_str = self.memory_manager.memory.to_context_string(max_turns=6)

            # Get answer (pass intent)
            if hasattr(self.rag_engine, "generate_response"):
                response = self.rag_engine.generate_response(rewritten_query, original_query=user_query, history=history_str, intent=intent)
            else:
                raw_resp = self.rag_engine.answer_question(rewritten_query)
                response = {
                    "answer": raw_resp.get("answer", ""),
                    "sources": [s.get("source_document") for s in raw_resp.get("sources", []) if s.get("source_document")],
                    "confidence": 1.0
                }
            
            # Update memory history
            self.memory_manager.memory.add_turn(user_query, response.get("answer", ""))
            
            # Update state — use original user_query so topic keywords are detectable
            self.memory_manager.update_state(
                user_query=user_query, 
                answer=response.get("answer", ""), 
                sources=response.get("sources", [])
            )
            
            return response
        except Exception as e:
            logger = logging.getLogger("rag_engine")
            logger.exception(e)
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
