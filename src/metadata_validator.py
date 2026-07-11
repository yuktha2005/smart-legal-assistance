import re
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class MetadataValidator:
    """Validates and enriches metadata for document chunks."""
    
    def __init__(self):
        self.stats = {
            "total": 0,
            "validated": 0,
            "discarded": 0,
            "missing_fields": {}
        }
        self.key_terms = [
            "theft", "extortion", "robbery", "dacoity", "murder", "homicide", 
            "assault", "kidnapping", "abduction", "rape", "defamation", 
            "forgery", "cheating", "mischief", "trespass", "conspiracy", "unlawful assembly"
        ]

    def validate_and_enrich(self, chunk: dict) -> dict | None:
        """
        Validates a chunk's metadata. 
        Returns enriched chunk if valid, else returns None if any required field is missing.
        """
        self.stats["total"] += 1
        text = str(chunk.get("text", ""))
        
        # 1. IPC Section
        ipc = chunk.get("ipc_section")
        if not ipc:
            ipc_match = re.search(r"\b(?:IPC|Section|Sec)\s*(\d+[A-Z]*)\b", text, re.IGNORECASE)
            ipc = ipc_match.group(1).upper() if ipc_match else "N/A"
        chunk["ipc_section"] = str(ipc).strip()
        
        # 2. BNS Section
        bns = chunk.get("bns_section")
        if not bns:
            bns_match = re.search(r"\bBNS\s*(?:Section|Sec)?\s*(\d+[A-Z]*)\b", text, re.IGNORECASE)
            bns = bns_match.group(1).upper() if bns_match else "N/A"
        chunk["bns_section"] = str(bns).strip()
        
        # 3. Crime Name
        crime = chunk.get("crime_name") or chunk.get("crime")
        if not crime:
            crime_match = re.search(r"(?i)(?:Crime|Offence|Offense):\s*(.*?)(?:\n|$)", text)
            if crime_match:
                crime = crime_match.group(1).strip()
            else:
                for term in self.key_terms:
                    if re.search(r"\b" + term + r"\b", text, re.IGNORECASE):
                        crime = term.title()
                        break
        chunk["crime_name"] = str(crime or "").strip()
        
        # 4. Description
        desc = chunk.get("description")
        if not desc:
            # Fallback to text itself if not explicitly structured
            desc = text.strip()
        chunk["description"] = str(desc).strip()
        
        # 5. Punishment
        pun = chunk.get("punishment")
        if not pun:
            # Extract sentence with punishment keywords
            sentences = re.split(r"(?<=[\.!?])\s+", text)
            punishment_sents = [s.strip() for s in sentences if any(w in s.lower() for w in ["punish", "imprison", "fine", "death penalty", "penalty", "sentence", "jail"])]
            if punishment_sents:
                pun = " ".join(punishment_sents)
            else:
                pun = ""
        chunk["punishment"] = str(pun).strip()
        
        # 6. Related Sections
        rel = chunk.get("related_sections")
        if not rel:
            all_sections = re.findall(r"\b(?:Section|Sec|IPC|BNS)\s*(\d+[A-Z]*)\b", text, re.IGNORECASE)
            unique_secs = set(s.upper() for s in all_sections)
            unique_secs.discard(chunk["ipc_section"])
            unique_secs.discard(chunk["bns_section"])
            rel = ", ".join(unique_secs) if unique_secs else "N/A"
        chunk["related_sections"] = str(rel).strip()
        
        # 7. Source Document
        src = chunk.get("source_document") or chunk.get("doc_source") or "unknown"
        chunk["source_document"] = str(src).strip()
        
        # 8. Page Number
        page = chunk.get("page_number")
        try:
            page = int(page)
        except (ValueError, TypeError):
            page = 1
        chunk["page_number"] = page
        
        # 9. Keywords
        kws = chunk.get("keywords")
        if not kws or not isinstance(kws, list):
            kws_set = set()
            for term in self.key_terms:
                if re.search(r"\b" + term + r"\b", text, re.IGNORECASE):
                    kws_set.add(term)
            if chunk["ipc_section"] != "N/A": kws_set.add(f"IPC {chunk['ipc_section']}")
            if chunk["bns_section"] != "N/A": kws_set.add(f"BNS {chunk['bns_section']}")
            kws = list(kws_set)
        chunk["keywords"] = kws
        
        # 10. Aliases
        aliases = chunk.get("aliases")
        if not aliases or not isinstance(aliases, list):
            aliases = []
            if chunk["crime_name"]:
                aliases.append(chunk["crime_name"].lower())
        chunk["aliases"] = aliases
        
        # 11. Complete Text
        chunk["complete_text"] = (
            f"Crime: {chunk['crime_name']}\n"
            f"IPC Section: {chunk['ipc_section']}\n"
            f"BNS Section: {chunk['bns_section']}\n"
            f"Description: {chunk['description']}\n"
            f"Punishment: {chunk['punishment']}\n"
            f"Related Sections: {chunk['related_sections']}\n"
            f"Source: {chunk['source_document']}\n"
            f"Page Number: {chunk['page_number']}"
        ).strip()
        
        # Validation checks
        required_fields = [
            "crime_name", "aliases", "keywords", "ipc_section", "bns_section", 
            "description", "punishment", "related_sections", "source_document", 
            "page_number", "complete_text"
        ]
        
        missing_fields = []
        for f in required_fields:
            if f not in chunk or chunk[f] is None:
                missing_fields.append(f)
            elif isinstance(chunk[f], str) and not chunk[f].strip():
                missing_fields.append(f)
            elif isinstance(chunk[f], list) and not chunk[f]:
                missing_fields.append(f)
                
        if missing_fields:
            for f in missing_fields:
                self.stats["missing_fields"][f] = self.stats["missing_fields"].get(f, 0) + 1
            logger.warning(f"Discarding chunk {chunk.get('chunk_id')} from {chunk.get('source_document')}. Missing: {missing_fields}")
            self.stats["discarded"] += 1
            return None
            
        self.stats["validated"] += 1
        return chunk
        
    def print_report(self):
        """Prints a summary of metadata coverage."""
        print("\n=== Metadata Validation Report ===")
        print(f"Total chunks evaluated: {self.stats['total']}")
        print(f"Successfully validated & kept: {self.stats['validated']} ({(self.stats['validated']/max(1, self.stats['total']))*100:.1f}%)")
        print(f"Discarded (missing fields): {self.stats['discarded']} ({(self.stats['discarded']/max(1, self.stats['total']))*100:.1f}%)")
        if self.stats["missing_fields"]:
            print("Missing occurrences per field:")
            for field, count in self.stats["missing_fields"].items():
                print(f"  - {field}: {count}")
        print("==================================\n")
