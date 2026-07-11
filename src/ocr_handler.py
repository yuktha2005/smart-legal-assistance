"""
OCR Handler Module
==================
Handle optical character recognition for scanned PDF documents.

Provides fallback OCR processing when PyMuPDF extraction fails or produces poor results.

Uses:
- Pytesseract: OCR engine wrapper
- Pillow: Image processing
- PyMuPDF: Convert PDF pages to images

Functions:
    - OCRHandler: Main OCR orchestration
"""

import fitz
import pytesseract
from PIL import Image
import numpy as np
from typing import List, Dict, Optional, Tuple
import io
import warnings

warnings.filterwarnings('ignore')


class OCRHandler:
    """
    Handle optical character recognition for scanned or image-heavy PDFs.
    
    Falls back to OCR when:
    - PyMuPDF extraction produces minimal text
    - Document appears to be image-based
    - Text quality is poor
    """
    
    def __init__(self, language: str = "eng", 
                 tesseract_path: Optional[str] = None):
        """
        Initialize OCR handler.
        
        Args:
            language (str): Tesseract language code (eng, spa, fra, etc.)
            tesseract_path (str, optional): Path to tesseract executable
        """
        self.language = language
        
        # Configure pytesseract if path provided
        if tesseract_path:
            pytesseract.pytesseract.pytesseract_cmd = tesseract_path
        
        self._verify_tesseract()
    
    def _verify_tesseract(self) -> bool:
        """
        Verify Tesseract is installed.
        
        Returns:
            bool: True if available
        """
        try:
            pytesseract.get_tesseract_version()
            print("✓ Tesseract OCR available")
            return True
        except Exception as e:
            print(f"⚠️ Tesseract not found: {e}")
            print("   Install: https://github.com/UB-Mannheim/tesseract/wiki")
            return False
    
    def is_scanned_pdf(self, pdf_path: str, sample_pages: int = 3) -> Tuple[bool, float]:
        """
        Detect if PDF is scanned (mostly images).
        
        Args:
            pdf_path (str): Path to PDF file
            sample_pages (int): Number of pages to sample
            
        Returns:
            Tuple[bool, float]: (is_scanned, text_ratio)
        """
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_to_check = min(sample_pages, total_pages)
        
        total_extracted = 0
        total_possible = 0
        
        for page_num in range(pages_to_check):
            page = doc[page_num]
            
            # Get extracted text
            text = page.get_text()
            extracted_chars = len(text.strip())
            
            # Estimate page size (rough approximation)
            page_height = page.rect.height
            page_width = page.rect.width
            estimated_capacity = (page_height * page_width) / 100  # Rough estimate
            
            total_extracted += extracted_chars
            total_possible += int(estimated_capacity)
        
        doc.close()
        
        # Calculate text ratio
        text_ratio = total_extracted / max(total_possible, 1)
        is_scanned = text_ratio < 0.1  # Less than 10% extracted
        
        return is_scanned, text_ratio
    
    def extract_text_with_ocr(self, pdf_path: str,
                             page_nums: Optional[List[int]] = None) -> List[str]:
        """
        Extract text from PDF using OCR.
        
        Args:
            pdf_path (str): Path to PDF file
            page_nums (List[int], optional): Specific pages to extract
            
        Returns:
            List[str]: Extracted text per page
        """
        print(f"🔍 Starting OCR extraction from {pdf_path}")
        
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        if page_nums is None:
            page_nums = list(range(total_pages))
        
        page_texts = []
        
        for page_num in page_nums:
            if page_num >= total_pages:
                continue
            
            try:
                page = doc[page_num]
                
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                img_data = pix.tobytes("ppm")
                image = Image.open(io.BytesIO(img_data))
                
                # Run OCR
                text = pytesseract.image_to_string(image, lang=self.language)
                page_texts.append(text)
                
                print(f"   Page {page_num + 1}/{len(page_nums)}: {len(text)} chars extracted")
            
            except Exception as e:
                print(f"   ⚠️ Error on page {page_num + 1}: {e}")
                page_texts.append("")
        
        doc.close()
        print(f"✓ OCR extraction complete: {len(page_texts)} pages")
        
        return page_texts
    
    def hybrid_extraction(self, pdf_path: str,
                         ocr_threshold: float = 0.1) -> Dict:
        """
        Try PyMuPDF first, fallback to OCR if needed.
        
        Args:
            pdf_path (str): Path to PDF file
            ocr_threshold (float): Ratio below which to trigger OCR
            
        Returns:
            Dict: Extraction results with method used
        """
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        # Try PyMuPDF first
        print(f"📄 Attempting PyMuPDF extraction...")
        pdfminer_texts = []
        text_extracted = 0
        
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text()
            pdfminer_texts.append(text)
            text_extracted += len(text.strip())
        
        doc.close()
        
        # Check if we need OCR
        text_ratio = text_extracted / max(total_pages * 1000, 1)  # Rough estimate
        
        if text_ratio < ocr_threshold:
            print(f"   Text ratio: {text_ratio:.2%} (below {ocr_threshold:.0%} threshold)")
            print(f"   Switching to OCR...")
            
            ocr_texts = self.extract_text_with_ocr(pdf_path)
            
            return {
                "method": "ocr",
                "texts": ocr_texts,
                "text_ratio": text_ratio,
                "fallback_used": True
            }
        else:
            print(f"   Text ratio: {text_ratio:.2%} - sufficient quality")
            
            return {
                "method": "pdfminer",
                "texts": pdfminer_texts,
                "text_ratio": text_ratio,
                "fallback_used": False
            }
    
    def get_ocr_confidence(self, image_path: str = None,
                          image: Image.Image = None) -> Dict:
        """
        Get OCR confidence metrics for an image.
        
        Args:
            image_path (str, optional): Path to image file
            image (PIL.Image, optional): Image object
            
        Returns:
            Dict: Confidence data
        """
        if image is None:
            image = Image.open(image_path)
        
        # Get OCR data with confidence
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT,
                                        lang=self.language)
        
        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
        
        if not confidences:
            avg_confidence = 0
            max_confidence = 0
        else:
            avg_confidence = np.mean(confidences)
            max_confidence = np.max(confidences)
        
        return {
            "avg_confidence": float(avg_confidence),
            "max_confidence": float(max_confidence),
            "words_detected": len([c for c in confidences if c > 0]),
            "low_confidence_words": len([c for c in confidences if c < 50])
        }
    
    def preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Processed image
        """
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize (if too small)
        if image.width < 400:
            scale = 400 / image.width
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Apply contrast enhancement
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Apply sharpness enhancement
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        return image
