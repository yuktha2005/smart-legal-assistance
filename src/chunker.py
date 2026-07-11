import re

class Chunker:
    def __init__(self, chunk_size: int = 200, overlap: int = 50):
        # We keep the init signature to avoid breaking callers, 
        # but we ignore chunk_size and overlap for 376-series section matching.
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, filename: str, pages_text: list[dict]) -> list[dict]:
        """
        Splits text from pages into one chunk per 376-series section.
        """
        chunks = []
        chunk_id_counter = 0
        current_section = "Unknown"

        # Match "376A", "376AB", etc.
        section_pattern = re.compile(r'\b(376[A-Z]*)\b')

        current_text_buffer = []
        current_page_num = 1

        for page_data in pages_text:
            page_num = page_data["page_number"]
            text = page_data["text"]
            
            # Split the text around the 376-series section markers
            parts = re.split(r'(\b376[A-Z]*\b)', text)
            
            i = 0
            while i < len(parts):
                part = parts[i]
                if section_pattern.fullmatch(part.strip()):
                    # A new 376-series section is found
                    # Flush the current buffer as a chunk for the PREVIOUS section
                    if current_text_buffer:
                        chunk_text = " ".join(current_text_buffer).strip()
                        if chunk_text:
                            chunks.append({
                                "chunk_id": f"{filename}_p{current_page_num}_c{chunk_id_counter}",
                                "source_document": filename,
                                "page_number": current_page_num,
                                "text": chunk_text,
                                "section_number": current_section
                            })
                            chunk_id_counter += 1
                        current_text_buffer = []
                    
                    # Update current section and page context
                    current_section = part.strip().upper()
                    current_page_num = page_num
                    
                    # We might want to keep the section heading in the text
                    current_text_buffer.append(f"Section {current_section}")
                else:
                    if part.strip():
                        current_text_buffer.append(part.strip())
                        current_page_num = page_num
                i += 1

        # Flush the final buffer
        if current_text_buffer:
            chunk_text = " ".join(current_text_buffer).strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": f"{filename}_p{current_page_num}_c{chunk_id_counter}",
                    "source_document": filename,
                    "page_number": current_page_num,
                    "text": chunk_text,
                    "section_number": current_section
                })

        return chunks
