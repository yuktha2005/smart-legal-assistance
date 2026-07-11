"""Smart Legal Assistance - Phase 2 Module: Source Formatter

Formats sources for the final response.

Desired format:
Sources:
• criminal_cases.pdf (Chunk 13)
Similarity: 94%
"""

from __future__ import annotations

from typing import Dict, List


class SourceFormatter:
    def format(self, sources: List[Dict]) -> str:
        if not sources:
            return "Sources:\n• (none)"

        lines: List[str] = ["Sources:"]
        for s in sources:
            doc = s.get("source_document") or "unknown"
            chunk_id = s.get("chunk_id")
            sim = s.get("similarity")
            sim_pct = ""
            if sim is not None:
                try:
                    # similarity from VectorStore is in [0..1]; format as percent
                    sim_pct = f" (Similarity: {float(sim)*100:.0f}%)"
                except Exception:
                    sim_pct = ""
            lines.append(f"• {doc} (Chunk {chunk_id}){sim_pct}")
        return "\n".join(lines)

