"""
Document Chunking Module
========================
Break documents into manageable semantic chunks with metadata preservation.

Client requirement (legal RAG):
- Provide multi-strategy chunking (Section Detection -> Paragraph -> Sentence -> Sliding Window).
- Never split a legal record across chunks.
- Extract basic metadata from unstructured text if possible.
"""

from __future__ import annotations

from typing import List, Dict, Tuple
import re


class DocumentChunker:
    """Break documents into semantic chunks for processing."""

    def __init__(self, chunk_size: int = 1000, overlap_size: int = 200, separator: str = "\n\n"):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.separator = separator
        print("Initialized Semantic Document Chunker (multi-strategy fallback)")

    def _extract_fields_from_text(self, text: str, filename: str) -> dict:
        """Extract structured fields from unstructured text chunk."""
        # Enforce all 11 keys in the output record dictionary
        record = {
            "crime_name": "",
            "aliases": [],
            "keywords": [],
            "ipc_section": "N/A",
            "bns_section": "N/A",
            "description": "",
            "punishment": "",
            "related_sections": "N/A",
            "source_document": filename,
            "page_number": 1,
            "complete_text": ""
        }
        
        # Regex to find sections
        ipc_match = re.search(r"\b(?:IPC|Section|Sec)\s*(\d+[A-Z]*)\b", text, re.IGNORECASE)
        if ipc_match:
            record["ipc_section"] = ipc_match.group(1).upper()
            
        bns_match = re.search(r"\bBNS\s*(?:Section|Sec)?\s*(\d+[A-Z]*)\b", text, re.IGNORECASE)
        if bns_match:
            record["bns_section"] = bns_match.group(1).upper()
            
        # Try to find a crime name
        crime_keywords = [
            "theft", "extortion", "robbery", "dacoity", "murder", "homicide", 
            "assault", "kidnapping", "abduction", "rape", "defamation", 
            "forgery", "cheating", "mischief", "trespass", "conspiracy", "common intention", "rioting",
            "criminal breach of trust", "dishonest misappropriation"
        ]
        
        # Look for headings: first line if short
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines and len(lines[0]) < 80 and any(kw in lines[0].lower() for kw in crime_keywords):
            record["crime_name"] = lines[0]
        else:
            # Match first keyword found in the chunk
            for kw in crime_keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", text, re.IGNORECASE):
                    record["crime_name"] = kw.title()
                    break
            if not record["crime_name"] and lines:
                record["crime_name"] = lines[0][:60]
                
        # Description: usually the chunk text itself
        record["description"] = text.strip()
        
        # Punishment: search for sentence containing punishment terms
        sentences = re.split(r"(?<=[\.!?])\s+", text)
        punishment_sents = []
        for s in sentences:
            if any(w in s.lower() for w in ["punish", "imprison", "fine", "death penalty", "penalty", "sentence", "jail"]):
                punishment_sents.append(s.strip())
        if punishment_sents:
            record["punishment"] = " ".join(punishment_sents)
        else:
            record["punishment"] = "N/A"
            
        # Related sections: search for other numbers
        all_sections = re.findall(r"\b(?:Section|Sec|IPC|BNS)\s*(\d+[A-Z]*)\b", text, re.IGNORECASE)
        unique_secs = set(s.upper() for s in all_sections)
        # Remove current section
        unique_secs.discard(record["ipc_section"])
        unique_secs.discard(record["bns_section"])
        if unique_secs:
            record["related_sections"] = ", ".join(unique_secs)
            
        # Keywords and aliases
        kws = set()
        for kw in crime_keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text, re.IGNORECASE):
                kws.add(kw)
        if record["ipc_section"] != "N/A": kws.add(f"IPC {record['ipc_section']}")
        if record["bns_section"] != "N/A": kws.add(f"BNS {record['bns_section']}")
        record["keywords"] = list(kws)
        
        if record["crime_name"]:
            record["aliases"] = [record["crime_name"].lower()]
            
        return record

    def chunk_document(self, text: str, doc_id: str = "unknown", doc_source: str = "unknown") -> List[Dict]:
        """Chunk a document using the strict fallback strategies hierarchy."""
        cleaned_text = text.strip()
        if not cleaned_text:
            return []

        chunks_data = []

        # Strategy 1: Section Detection
        # Match pattern: Section/Sec/IPC number followed by dot/space, or lines starting with Section number
        section_pattern = r"(?i)(?=\n\s*(?:Section|Sec|IPC)?\s*\d+[A-Z]*\.)"
        sections = re.split(section_pattern, "\n" + cleaned_text)
        sections = [s.strip() for s in sections if s.strip()]
        
        if len(sections) > 1:
            print(f"  -> Semantic Chunker: detected {len(sections)} sections")
            for sec_text in sections:
                meta = self._extract_fields_from_text(sec_text, doc_source)
                meta["chunk_id"] = len(chunks_data)
                meta["text"] = sec_text
                chunks_data.append(meta)
            return chunks_data

        # Fallback 1: Paragraph Chunking
        paragraphs = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            print(f"  -> Semantic Chunker: falling back to Paragraph Chunking ({len(paragraphs)} paragraphs)")
            for p_text in paragraphs:
                meta = self._extract_fields_from_text(p_text, doc_source)
                meta["chunk_id"] = len(chunks_data)
                meta["text"] = p_text
                chunks_data.append(meta)
            return chunks_data

        # Fallback 2: Sentence Chunking
        sentences = re.split(r"(?<=[\.!?])\s+", cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 1:
            print(f"  -> Semantic Chunker: falling back to Sentence Chunking ({len(sentences)} sentences)")
            current_chunk = ""
            for s in sentences:
                if len(current_chunk) + len(s) > self.chunk_size and current_chunk:
                    meta = self._extract_fields_from_text(current_chunk, doc_source)
                    meta["chunk_id"] = len(chunks_data)
                    meta["text"] = current_chunk
                    chunks_data.append(meta)
                    current_chunk = s
                else:
                    current_chunk = (current_chunk + " " + s).strip()
            if current_chunk:
                meta = self._extract_fields_from_text(current_chunk, doc_source)
                meta["chunk_id"] = len(chunks_data)
                meta["text"] = current_chunk
                chunks_data.append(meta)
            return chunks_data

        # Fallback 3: Sliding Window
        print("  -> Semantic Chunker: falling back to Sliding Window")
        words = cleaned_text.split()
        chunk_words = 150  # approx 1000 chars
        overlap_words = 30
        
        i = 0
        while i < len(words):
            chunk_txt = " ".join(words[i:i + chunk_words])
            meta = self._extract_fields_from_text(chunk_txt, doc_source)
            meta["chunk_id"] = len(chunks_data)
            meta["text"] = chunk_txt
            chunks_data.append(meta)
            i += (chunk_words - overlap_words)
            
        return chunks_data

    def chunk_documents(self, documents: List[Tuple[str, str, str]]) -> List[Dict]:
        """Chunk a list of (text, doc_id, doc_source) tuples."""
        all_chunks: List[Dict] = []
        for text, doc_id, doc_source in documents:
            all_chunks.extend(self.chunk_document(text, doc_id=doc_id, doc_source=doc_source))
        print(f"✓ Chunked {len(documents)} documents → {len(all_chunks)} semantic chunks")
        return all_chunks
