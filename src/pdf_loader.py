import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

class PDFLoader:
    def __init__(self, fallback_ocr: bool = True):
        self.fallback_ocr = fallback_ocr

    def extract_text(self, file_path: str) -> list[dict]:
        """
        Extracts text from a PDF file. Returns a list of dictionaries,
        each containing 'page_number' and 'text'.
        """
        doc = fitz.open(file_path)
        pages_text = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()

            # Fallback to OCR if page has no text but has images
            if not text and self.fallback_ocr:
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                text = pytesseract.image_to_string(img).strip()

            if text:
                pages_text.append({
                    "page_number": page_num + 1,
                    "text": text
                })

        return pages_text
