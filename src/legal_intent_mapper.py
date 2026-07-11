import re

class LegalIntentMapper:
    def __init__(self):
        self.mapping = {
            "rape punishment": "section 376",
            "gang rape": "section 376DA",
            "child rape": "section 376AB",
            "rape": "section 376",
            "murder": "section 302",
            "kidnapping": "section 363"
        }
        
    def map_intent(self, query: str) -> str:
        # Sort keys by length descending to match longer phrases first
        mapped_query = query
        for key in sorted(self.mapping.keys(), key=len, reverse=True):
            # Replace whole word matches only, ignoring case
            mapped_query = re.sub(r'\b' + re.escape(key) + r'\b', self.mapping[key], mapped_query, flags=re.IGNORECASE)
        return mapped_query
