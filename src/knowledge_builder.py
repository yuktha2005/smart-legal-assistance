import json
import os
from src.table_extractor import TableExtractor

class KnowledgeBuilder:
    def __init__(self):
        self.extractor = TableExtractor()
        
    def build_knowledge_base(self, pdf_path: str, output_path: str = "data/legal_knowledge.json"):
        records = self.extractor.extract_from_pdf(pdf_path)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4)
            
        print(f"Extracted {len(records)} records. Saved to {output_path}")
        return records
