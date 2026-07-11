import logging
import os
import re
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

from src.context_builder import ContextBuilder
from src.hallucination_guard import HallucinationGuard
from src.ollama_generator import OllamaGenerator
from src.retriever import Retriever
from src.intent_classifier import IntentClassifier

os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("rag_engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(os.path.join("logs", "app.log"))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class EnhancedRAGEngine:
    def __init__(self, retriever: Retriever, model_name: str = "qwen3:0.6b"):
        self.retriever = retriever
        self.answer_generator = OllamaGenerator(model_name=model_name)
        self.context_builder = ContextBuilder()
        self.hallucination_guard = HallucinationGuard()
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.intent_classifier = IntentClassifier()

    def _safe_print(self, text):
        try:
            print(text)
        except UnicodeEncodeError:
            print(str(text).encode('ascii', 'replace').decode('ascii'))

    def generate_response(self, query: str, original_query: str = "", history: str = "", intent: str = None) -> dict:
        try:
            logger.info(f"--- New Interaction ---")
            if original_query:
                logger.info(f"Original Query: {original_query}")
            logger.info(f"Rewritten Query: {query}")
            
            # Step 1a: Intent Classification
            if not intent:
                intent = self.intent_classifier.get_intent(original_query if original_query else query)
            logger.info(f"Detected Intent: {intent}")
            
            # Step 1b: Hybrid Retrieval (top 20) with intent boosting
            chunks = self.retriever.retrieve(query, top_k=20, intent=intent)
            retrieved_chunks_before_rerank = [c.copy() for c in chunks]

            if not chunks:
                logger.info("No chunks retrieved.")
                fallback = "I couldn't find sufficient information in the uploaded legal documents."
                
                # Debugging print in case of no retrieval
                self._safe_print("\n" + "="*50)
                self._safe_print("RETRIEVAL DEBUGGING INFO")
                self._safe_print("="*50)
                self._safe_print(f"Original Query: {original_query if original_query else query}")
                self._safe_print("↓")
                self._safe_print(f"Rewritten Query: {query}")
                self._safe_print("↓")
                self._safe_print(f"Intent: {intent}")
                self._safe_print("↓")
                self._safe_print("Retrieved Chunks: None")
                self._safe_print("↓")
                self._safe_print("Reranked Chunks: None")
                self._safe_print("↓")
                self._safe_print("Context Sent to Ollama: None")
                self._safe_print("↓")
                self._safe_print("Final Response:")
                self._safe_print(fallback)
                self._safe_print("="*50 + "\n")
                
                return {"answer": fallback, "sources": [], "confidence": 0.0}

            # Step 2: Cross-Encoder Reranking
            pairs = [[query, c.get("complete_text", c.get("text", ""))] for c in chunks]
            scores = self.cross_encoder.predict(pairs)

            for i, score in enumerate(scores):
                chunks[i]["cross_score"] = float(score)

            chunks.sort(key=lambda x: x.get("cross_score", -999), reverse=True)
            # Take only the top-ranked best legal records (up to 5 records)
            valid_chunks = chunks[:5]

            # Step 3: Context Builder (merges duplicate records across PDFs)
            context = self.context_builder.build(valid_chunks)

            # Step 4: Deduplicate and format sources
            source_pages = {}
            for c in valid_chunks:
                src = c.get('source_document')
                page = c.get('page_number')
                if src:
                    if src not in source_pages:
                        source_pages[src] = []
                    # Keep track of unique page numbers
                    if page and page not in source_pages[src]:
                        source_pages[src].append(page)

            sources_parts = []
            for src, pages in source_pages.items():
                if pages:
                    pages_str = ", ".join(map(str, sorted(pages)))
                    sources_parts.append(f"• {src} (Page {pages_str})")
                else:
                    sources_parts.append(f"• {src}")
            sources_text = "\n".join(sources_parts)

            # Step 5: Answer generation
            answer_gen = self.answer_generator.generate_answer(query, context, history, sources=sources_text)
            answer = ""
            for chunk in answer_gen:
                answer += chunk

            if not answer or answer.strip() == "":
                answer = "I couldn't find sufficient information in the uploaded legal documents."

            # Step 6: Hallucination Guard
            top_score = valid_chunks[0].get("cross_score", 0) if valid_chunks else 0
            confidence = max([c.get("similarity", 0) for c in valid_chunks]) if valid_chunks else 0
            
            # adjust confidence calculation
            confidence_label = "High" if (confidence >= 0.70 or top_score > 0.0) else "Medium" if confidence >= 0.50 else "Low"
            guard_result = self.hallucination_guard.guard(answer, confidence_label, valid_chunks)

            final_answer = guard_result.final_answer
            if not final_answer or final_answer.strip() == "":
                final_answer = "I couldn't find sufficient information in the uploaded legal documents."
            else:
                # If the hallucination guard fell back to low_confidence_message, it will not have sources
                # Otherwise, ensure Sources header is clean and not duplicated
                if "Sources" not in final_answer:
                    if sources_text:
                        final_answer += f"\n\n### Sources\n{sources_text}"

            logger.info(f"Final Answer Generated")

            # REQUIRED debug logging print workflow for every query
            self._safe_print("\n" + "="*50)
            self._safe_print("RETRIEVAL DEBUGGING INFO")
            self._safe_print("="*50)
            self._safe_print(f"Original Query: {original_query if original_query else query}")
            self._safe_print("↓")
            self._safe_print(f"Rewritten Query: {query}")
            self._safe_print("↓")
            self._safe_print(f"Intent: {intent}")
            self._safe_print("↓")
            self._safe_print("Retrieved Chunks:")
            for i, c in enumerate(retrieved_chunks_before_rerank):
                self._safe_print(f"  [{i}] ID: {c.get('chunk_id')} | IPC: {c.get('ipc_section')} | BNS: {c.get('bns_section')} | Score: {c.get('similarity', 0):.2f} | Src: {c.get('source_document')}")
            self._safe_print("↓")
            self._safe_print("Reranked Chunks:")
            for i, c in enumerate(valid_chunks):
                self._safe_print(f"  [{i}] ID: {c.get('chunk_id')} | IPC: {c.get('ipc_section')} | BNS: {c.get('bns_section')} | Cross Score: {c.get('cross_score', 0):.2f} | Src: {c.get('source_document')}")
            self._safe_print("↓")
            self._safe_print("Context Sent to Ollama:")
            self._safe_print(context)
            self._safe_print("↓")
            self._safe_print("Final Response:")
            self._safe_print(final_answer)
            self._safe_print("="*50 + "\n")

            return {
                "answer": final_answer,
                "sources": list(source_pages.keys()),
                "confidence": confidence,
            }
        except Exception as e:
            logger.exception(e)
            fallback = f"Error: {str(e)}"
            self._safe_print("ANSWER: " + fallback)
            return {"answer": fallback, "sources": [], "confidence": 0.0}
