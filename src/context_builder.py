from __future__ import annotations

from typing import List, Dict, Any


class ContextBuilder:
    """Build structured, merged legal context for the LLM.

    Converts retrieved record chunks into a clean, consistent format:
      Crime: ...
      IPC Section: ...
      BNS Section: ...
      Description: ...
      Punishment: ...
      Source Document: ...
      Page Number: ...

    Removes duplicate records based on crime name and sections.
    """

    def build(self, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return ""

        seen_keys = set()
        unique_records = []
        
        for c in chunks:
            # Clean values for composite key construction
            crime = str(c.get("crime_name") or c.get("crime") or "").strip().lower()
            ipc = str(c.get("ipc_section") or "").strip().upper()
            bns = str(c.get("bns_section") or "").strip().upper()
            
            # Skip duplicates using composite key
            key = (crime, ipc, bns)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            
            unique_records.append(c)

        # Limit to top 5 unique records
        top_records = unique_records[:5]
        
        context_parts = []
        for r in top_records:
            crime = r.get("crime_name") or r.get("crime") or "N/A"
            ipc = r.get("ipc_section") or "N/A"
            bns = r.get("bns_section") or "N/A"
            desc = r.get("description") or "N/A"
            pun = r.get("punishment") or "N/A"
            src = r.get("source_document") or "N/A"
            page = r.get("page_number") or "N/A"
            
            block = (
                f"Crime: {crime}\n"
                f"IPC Section: {ipc}\n"
                f"BNS Section: {bns}\n"
                f"Description: {desc}\n"
                f"Punishment: {pun}\n"
                f"Source Document: {src}\n"
                f"Page Number: {page}"
            )
            context_parts.append(block)
            
        return "\n\n---\n\n".join(context_parts)
