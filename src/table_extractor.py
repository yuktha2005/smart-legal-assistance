import pdfplumber
import re
import fitz
import os

class TableExtractor:
    def __init__(self):
        # Map common header keywords to our schema fields
        self.header_mapping = {
            "ipc": "ipc_section",
            "section": "ipc_section",
            "sec": "ipc_section",
            "bns": "bns_section",
            "crime": "crime",
            "offense": "crime",
            "offence": "crime",
            "description": "description",
            "details": "description",
            "definition": "description",
            "criteria": "description",
            "victim": "description",
            "ingredient": "ingredients",
            "ingredients": "ingredients",
            "element": "ingredients",
            "punishment": "punishment",
            "penalty": "punishment",
            "sentence": "punishment",
            "case": "important_cases",
            "cases": "important_cases",
            "keyword": "keywords",
            "keywords": "keywords",
            "related": "related_sections",
            "cross": "related_sections",
        }
        # Sort header mapping keys by length descending to prioritize longer phrases (e.g. punishment over ipc)
        self.header_keys_sorted = sorted(self.header_mapping.keys(), key=len, reverse=True)

    def _identify_columns(self, header_row: list, file_path: str = "") -> dict:
        """Map column indices to schema fields based on header text."""
        is_bns_doc = "bns" in os.path.basename(file_path).lower() if file_path else False
        col_map = {}
        for i, cell in enumerate(header_row):
            if not cell:
                continue
            cell_lower = str(cell).lower().strip()
            
            # Check for combined IPC Section and Crime
            if ("ipc" in cell_lower or "section" in cell_lower or "sec" in cell_lower) and \
               ("crime" in cell_lower or "offense" in cell_lower or "offence" in cell_lower):
                col_map[i] = "ipc_section_and_crime"
            elif "bns" in cell_lower:
                col_map[i] = "bns_section"
            else:
                # Find the best match using sorted keys to prevent false substring matches (e.g. ipc in punishment)
                for key in self.header_keys_sorted:
                    if key in cell_lower:
                        field = self.header_mapping[key]
                        if field == "ipc_section" and is_bns_doc:
                            col_map[i] = "bns_section"
                        elif field == "ipc_section" and "bns" in cell_lower:
                            col_map[i] = "bns_section"
                        else:
                            col_map[i] = field
                        break
        return col_map

    def _extract_from_text_block(self, text: str, page_num: int) -> dict:
        """Parse unstructured compound text block into structured legal record fields."""
        record = {
            "ipc_section": "",
            "crime": "",
            "description": "",
            "ingredients": "",
            "punishment": "",
            "important_cases": "",
            "bns_section": "",
            "keywords": [],
            "related_sections": "",
            "page_number": page_num
        }
        
        # Regex to find IPC section
        ipc_match = re.search(r"\b(?:IPC|Section|Sec)\s*(\d+[A-Z]*)\b", text, re.IGNORECASE)
        if ipc_match:
            record["ipc_section"] = ipc_match.group(1).upper()
            
        # Regex to find BNS section
        bns_match = re.search(r"\bBNS\s*(?:Section|Sec)?\s*(\d+[A-Z]*)\b", text, re.IGNORECASE)
        if bns_match:
            record["bns_section"] = bns_match.group(1).upper()
            
        # Try to find punishment (sentence containing punishment words)
        sentences = re.split(r"(?<=[\.!?])\s+", text)
        punishment_sents = [s.strip() for s in sentences if any(w in s.lower() for w in ["punish", "imprison", "fine", "death penalty", "penalty", "sentence", "jail"])]
        if punishment_sents:
            record["punishment"] = " ".join(punishment_sents)
            
        # Extract crime name
        crime_keywords = [
            "theft", "extortion", "robbery", "dacoity", "murder", "homicide", 
            "assault", "kidnapping", "abduction", "rape", "defamation", 
            "forgery", "cheating", "mischief", "trespass", "conspiracy", "common intention", "rioting",
            "criminal breach of trust", "dishonest misappropriation", "public servant disobeying law"
        ]
        
        for kw in crime_keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text, re.IGNORECASE):
                record["crime"] = kw.title()
                break
                
        if not record["crime"]:
            first_line = text.split("\n")[0].strip()
            first_line_clean = re.sub(r"^\s*(?:Sec\.|Section|IPC|Sec)?\s*\d+[A-Z]*\b", "", first_line, flags=re.IGNORECASE).strip()
            record["crime"] = first_line_clean[:60] if first_line_clean else "Legal Entry"
            
        # Description: clean text minus section, BNS mapping and punishment
        desc_text = text
        if record["ipc_section"]:
            desc_text = re.sub(r"\b(?:IPC|Section|Sec)\s*" + re.escape(record["ipc_section"]) + r"\b", "", desc_text, flags=re.IGNORECASE)
        if record["bns_section"]:
            desc_text = re.sub(r"\bBNS\s*(?:Section|Sec)?\s*" + re.escape(record["bns_section"]) + r"\b", "", desc_text, flags=re.IGNORECASE)
        if record["punishment"]:
            desc_text = desc_text.replace(record["punishment"], "")
            
        record["description"] = re.sub(r"\s+", " ", desc_text).strip()
        
        return record

    def extract_from_pdf(self, file_path: str) -> list[dict]:
        """
        Extracts structured table rows from a PDF.
        Returns a list of dictionaries with mapped fields.
        """
        records = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    
                    if not tables:
                        continue
                        
                    for table in tables:
                        if not table or len(table) < 2:
                            continue
                            
                        # Find header (the row that maps to key columns)
                        header_idx = -1
                        max_cols = 0
                        best_col_map = {}
                        
                        for i, row in enumerate(table[:5]): # check first few rows for header
                            if not row:
                                continue
                            col_map = self._identify_columns(row, file_path)
                            # Count how many key columns we matched
                            key_cols_count = sum(1 for f in col_map.values() if f in ["ipc_section", "crime", "description", "punishment", "ipc_section_and_crime"])
                            if key_cols_count > max_cols:
                                max_cols = key_cols_count
                                header_idx = i
                                best_col_map = col_map
                                
                        if header_idx == -1 or max_cols == 0:
                            continue
                            
                        col_map = best_col_map
                            
                        # Process data rows and merge wrapped lines
                        cleaned_rows = []
                        current_category = ""
                        
                        sect_crime_indices = []
                        for idx, field in col_map.items():
                            if field in ["ipc_section", "bns_section", "crime", "ipc_section_and_crime"]:
                                sect_crime_indices.append(idx)
                                
                        for row in table[header_idx+1:]:
                            if not row or not any(row):
                                continue
                                
                            cells = [str(cell).strip() if cell is not None else "" for cell in row]
                            
                            # Check if it's a category header
                            non_empty_indices = [i for i, val in enumerate(cells) if val]
                            if len(non_empty_indices) == 1:
                                first_val = cells[non_empty_indices[0]]
                                if any(w in first_val.upper() for w in ["MATRIX", "OFFENCES", "CHAPTER", "PROVISIONS", "PROCEDURE", "PENAL OFFENCES"]):
                                    current_category = first_val
                                    continue
                                    
                                # If it's a long text block, parse it directly as unstructured record
                                if len(first_val) > 100:
                                    parsed_rec = self._extract_from_text_block(first_val, page_num)
                                    if current_category:
                                        parsed_rec["keywords"].append(current_category.strip())
                                    records.append(parsed_rec)
                                    continue
                                    
                                if len(first_val) > 0:
                                    continue
                                    
                            # Determine if this row is a continuation of the last row
                            is_continuation = False
                            if cleaned_rows:
                                if sect_crime_indices and all(not cells[idx] for idx in sect_crime_indices):
                                    if any(cells[idx] for idx in col_map.keys() if idx not in sect_crime_indices):
                                        is_continuation = True
                                        
                            if is_continuation:
                                prev_row = cleaned_rows[-1]
                                for idx in range(len(cells)):
                                    if cells[idx]:
                                        prev_row[idx] = (prev_row[idx] + " " + cells[idx]).strip()
                            else:
                                cleaned_rows.append(cells)
                                
                        # Build record dictionaries
                        for clean_cells in cleaned_rows:
                            record = {
                                "ipc_section": "",
                                "crime": "",
                                "description": "",
                                "ingredients": "",
                                "punishment": "",
                                "important_cases": "",
                                "bns_section": "",
                                "keywords": [],
                                "related_sections": "",
                                "page_number": page_num
                            }
                            
                            if current_category:
                                record["keywords"].append(current_category.strip())
                                
                            has_data = False
                            for i, cell_val in enumerate(clean_cells):
                                if i in col_map and cell_val:
                                    field = col_map[i]
                                    if field == "keywords":
                                        record[field] = [k.strip() for k in cell_val.split(",") if k.strip()]
                                    elif field == "ipc_section_and_crime":
                                        sec_match = re.match(r"^\s*(?:Sec\.|Section|IPC|Sec)?\s*(\d+[A-Z]*)(?:\s*\(.*?\))?", cell_val, re.IGNORECASE)
                                        if sec_match:
                                            record["ipc_section"] = sec_match.group(1).upper()
                                            record["crime"] = cell_val[sec_match.end():].strip().lstrip("-").strip()
                                        else:
                                            record["crime"] = cell_val
                                    else:
                                        record[field] = cell_val
                                    has_data = True
                                    
                            if has_data:
                                records.append(record)
        except Exception as e:
            print(f"Error in pdfplumber table extraction for {file_path}: {e}")
            
        return records
