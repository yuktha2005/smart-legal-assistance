import re
from typing import Dict, Any, List

class EntityTracker:
    def __init__(self):
        self.conversation_context = {}

    def extract_entities(self, text: str) -> dict:
        entities = {}
        
        # Extract ANY section number (IPC or BNS)
        section_matches = re.findall(r'(?i)\b(?:section|sec\.?|ipc|bns)\s*(\d+[A-Z]*)\b', text)
        if section_matches:
            entities['section_number'] = section_matches[0].upper()
            entities['all_sections'] = [s.upper() for s in section_matches]
            
        # Standalone sections
        if not section_matches:
            standalone = re.findall(r'\b(\d{2,3}[A-Z]*)\b', text)
            if standalone:
                entities['possible_sections'] = [s.upper() for s in standalone]
            
        # Extract document (e.g., something.pdf)
        doc_match = re.search(r'\b(\w+\.pdf)\b', text, re.IGNORECASE)
        if doc_match:
            entities['document_name'] = doc_match.group(1)
            
        # Track crimes
        # We can also track crimes dynamically if passed from external metadata, but simple extraction works.
        
        # Update conversation context
        self.conversation_context.update(entities)
        
        return entities
