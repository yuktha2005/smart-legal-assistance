import difflib
import re
import os
import pickle

class LegalTermMapper:
    def __init__(self, metadata_path=None):
        self.mapping = {}
        self.metadata_path = metadata_path
        self._load_dynamic_mapping()
        
    def _load_dynamic_mapping(self):
        """Dynamically load mappings from the vector store metadata."""
        paths_to_try = [self.metadata_path, 'data/metadata.pkl', 'models/metadata.pkl', 'data/vector_store/metadata.pkl', 'data/faiss/metadata.pkl']
        
        metadata_path = None
        for path in paths_to_try:
            if path and os.path.exists(path):
                metadata_path = path
                break
                
        if metadata_path:
            try:
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                for record in metadata:
                    if isinstance(record, dict):
                        crime = record.get('crime_name', '').lower().strip()
                        ipc = record.get('ipc_section', '').strip()
                        
                        if crime and ipc and ipc != "N/A" and ipc.upper() != "N/A" and "not applicable" not in ipc.lower():
                            if crime not in self.mapping:
                                self.mapping[crime] = ipc
                            
                        # Handle aliases
                        for alias in record.get('aliases', []):
                            alias = alias.lower().strip()
                            if alias and ipc and ipc != "N/A" and ipc.upper() != "N/A" and "not applicable" not in ipc.lower() and alias not in self.mapping:
                                self.mapping[alias] = ipc
                                
                        generic_words = {
                            "punishment", "description", "section", "sec", "crime", "offence", "offense",
                            "penalty", "fine", "imprisonment", "imprison", "jail", "compensation", "injury",
                            "valuable", "security", "court", "judge", "police", "arrest", "warrant", "bail",
                            "trafficking", "repeat", "offenders", "subsequent", "conviction", "enhanced",
                            "code", "procedure", "criminal", "act", "prohibition", "certain", "operations",
                            "danger", "obstruction", "public", "way", "using", "use", "person", "property",
                            "age", "victim", "criteria", "matrix"
                        }
                        # Handle keywords mapping to concepts
                        for kw in record.get('keywords', []):
                            kw = kw.lower().strip()
                            if kw and kw not in self.mapping and not kw.startswith("ipc") and not kw.startswith("bns"):
                                if kw not in generic_words and len(kw) >= 4:
                                    if ipc and ipc != "N/A" and ipc.upper() != "N/A" and "not applicable" not in ipc.lower():
                                        self.mapping[kw] = ipc

                                    
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to load dynamic terms: {e}")
                
    def get_section(self, term: str) -> str:
        term_clean = term.lower().strip()
        if term_clean in self.mapping:
            return self.mapping[term_clean]
            
        if self.mapping:
            matches = difflib.get_close_matches(term_clean, self.mapping.keys(), n=1, cutoff=0.7)
            if matches:
                return self.mapping[matches[0]]
        return None

    def map_terms(self, query: str) -> str:
        mapped_query = query
        if not self.mapping:
            return mapped_query
            
        # Sort keys by length descending to match longer phrases first
        for key in sorted(self.mapping.keys(), key=len, reverse=True):
            if len(key) < 4: continue # skip very short acronyms to prevent bad mapping
            mapped_query = re.sub(r'\b' + re.escape(key) + r'\b', f"{key} (IPC Section {self.mapping[key]})", mapped_query, flags=re.IGNORECASE)
        return mapped_query
