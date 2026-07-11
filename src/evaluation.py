"""Smart Legal Assistance - Phase 2 Module 8

EvaluationReport
-----------------
Measures:
1) Retrieval performance
   - Top-1 Accuracy
   - Top-3 Accuracy
   - Average Similarity Score

2) RAG performance
   - Groundedness (answer sentences must appear in sources)
   - Context relevance (token overlap)
   - Hallucination detection (answer tokens not present in sources)

3) Summarizer performance
   - Compression ratio

Writes:
- reports/phase2_evaluation_report.md
- reports/phase2_evaluation_report.pdf

No external APIs.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

try:
    from reportlab.lib.pagesizes import letter  # type: ignore
    from reportlab.pdfgen import canvas  # type: ignore
except Exception:  # pragma: no cover
    letter = None
    canvas = None



logger = logging.getLogger(__name__)


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


class EvaluationReport:
    def __init__(self, output_md_path: str, output_pdf_path: str):
        self.output_md_path = output_md_path
        self.output_pdf_path = output_pdf_path

    def _groundedness(self, answer: str, sources_texts: List[str]) -> float:
        if not answer.strip():
            return 0.0
        sent_list = _split_sentences(answer)
        if not sent_list:
            return 0.0
        joined = "\n".join(sources_texts)
        grounded = sum(1 for s in sent_list if s in joined)
        return float(grounded / len(sent_list))

    def _context_relevance(self, question: str, sources_texts: List[str]) -> float:
        # token overlap metric
        q_tokens = set(re.findall(r"\b[a-zA-Z0-9_/-]{3,}\b", question.lower()))
        if not q_tokens:
            return 0.0
        src_tokens = set(re.findall(r"\b[a-zA-Z0-9_/-]{3,}\b", " ".join(sources_texts).lower()))
        return float(len(q_tokens & src_tokens) / len(q_tokens))

    def _hallucination_rate(self, answer: str, sources_texts: List[str]) -> float:
        ans_tokens = set(re.findall(r"\b[a-zA-Z0-9_/-]{3,}\b", answer.lower()))
        if not ans_tokens:
            return 0.0
        src_tokens = set(re.findall(r"\b[a-zA-Z0-9_/-]{3,}\b", " ".join(sources_texts).lower()))
        # hallucinated if token not in sources
        halluc = len(ans_tokens - src_tokens)
        return float(halluc / len(ans_tokens))

    def generate_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute metrics from a list of run results."""
        retrieval_top1 = 0
        retrieval_top3 = 0
        retrieval_sim_sum = 0.0
        retrieval_n = 0

        grounded_sum = 0.0
        context_rel_sum = 0.0
        halluc_sum = 0.0
        rag_n = 0

        # summarizer compression ratio
        compression_ratios = []

        for r in results:
            # Retrieval metrics assume r has expected_chunk_ids
            expected = set(r.get("expected_chunk_ids", []))
            retrieved = r.get("retrieved_chunks", [])
            retrieved_ids = [x.get("chunk_id") for x in retrieved]
            retrieved_ids = [i for i in retrieved_ids if i is not None]

            if expected:
                retrieval_n += 1
                if retrieved_ids[:1] and retrieved_ids[0] in expected:
                    retrieval_top1 += 1
                if any(i in expected for i in retrieved_ids[:3]):
                    retrieval_top3 += 1

            sims = [x.get("similarity") for x in retrieved]
            sims = [s for s in sims if s is not None]
            if sims:
                retrieval_sim_sum += float(sum(sims) / len(sims))

            # RAG metrics
            answer = r.get("answer", "")
            sources_texts = r.get("sources_texts", [])
            grounded_sum += self._groundedness(answer, sources_texts)
            context_rel_sum += self._context_relevance(r.get("question", ""), sources_texts)
            halluc_sum += self._hallucination_rate(answer, sources_texts)
            rag_n += 1

            # summarizer compression
            if "original_text" in r and "summary_text" in r and r["original_text"]:
                orig = r["original_text"]
                summ = r["summary_text"]
                if orig:
                    compression_ratios.append(float(len(summ) / len(orig)))

        metrics = {
            "retrieval": {
                "top_1_accuracy": float(retrieval_top1 / max(retrieval_n, 1)),
                "top_3_accuracy": float(retrieval_top3 / max(retrieval_n, 1)),
                "average_similarity_score": float(retrieval_sim_sum / max(retrieval_n, 1)),
            },
            "rag": {
                "groundedness": float(grounded_sum / max(rag_n, 1)),
                "context_relevance": float(context_rel_sum / max(rag_n, 1)),
                "hallucination_rate": float(halluc_sum / max(rag_n, 1)),
            },
            "summarizer": {
                "compression_ratio_avg": float(sum(compression_ratios) / max(len(compression_ratios), 1)),
            },
        }

        return metrics

    def write_md(self, metrics: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.output_md_path), exist_ok=True)
        md = [
            "# SMART LEGAL ASSISTANCE - Phase 2 Evaluation Report\n",
            "## Summary\n",
            "- Retrieval: Top-1 / Top-3 / Avg similarity\n",
            "- RAG: Groundedness / Context relevance / Hallucination rate\n",
            "- Summarizer: Compression ratio\n",
            "---\n",
            "## Retrieval Performance\n",
            f"- Top-1 Accuracy: {metrics['retrieval']['top_1_accuracy']:.3f}\n",
            f"- Top-3 Accuracy: {metrics['retrieval']['top_3_accuracy']:.3f}\n",
            f"- Average Similarity Score: {metrics['retrieval']['average_similarity_score']:.3f}\n",
            "---\n",
            "## RAG Performance\n",
            f"- Groundedness: {metrics['rag']['groundedness']:.3f}\n",
            f"- Context Relevance: {metrics['rag']['context_relevance']:.3f}\n",
            f"- Hallucination Detection (rate): {metrics['rag']['hallucination_rate']:.3f}\n",
            "---\n",
            "## Summarizer Performance\n",
            f"- Compression Ratio (avg): {metrics['summarizer']['compression_ratio_avg']:.3f}\n",
        ]

        with open(self.output_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))

    def write_pdf(self, metrics: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.output_pdf_path), exist_ok=True)
        c = canvas.Canvas(self.output_pdf_path, pagesize=letter)
        width, height = letter
        y = height - 50
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "SMART LEGAL ASSISTANCE - Phase 2 Evaluation Report")
        y -= 25
        c.setFont("Helvetica", 11)

        lines = [
            "Retrieval Performance:",
            f"  - Top-1 Accuracy: {metrics['retrieval']['top_1_accuracy']:.3f}",
            f"  - Top-3 Accuracy: {metrics['retrieval']['top_3_accuracy']:.3f}",
            f"  - Average Similarity Score: {metrics['retrieval']['average_similarity_score']:.3f}",
            "",
            "RAG Performance:",
            f"  - Groundedness: {metrics['rag']['groundedness']:.3f}",
            f"  - Context Relevance: {metrics['rag']['context_relevance']:.3f}",
            f"  - Hallucination Rate: {metrics['rag']['hallucination_rate']:.3f}",
            "",
            "Summarizer Performance:",
            f"  - Compression Ratio (avg): {metrics['summarizer']['compression_ratio_avg']:.3f}",
        ]

        for line in lines:
            c.drawString(50, y, line)
            y -= 16
            if y < 40:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 11)

        c.save()

    def run_and_write(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics = self.generate_report(results)
        self.write_md(metrics)
        self.write_pdf(metrics)
        return metrics

