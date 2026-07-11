"""
PDF Extractor Module
====================
Handles extraction of text and metadata from PDF documents using PyMuPDF (fitz).

Functions:
    - extract_text_from_pdf: Extract all text from a PDF file
    - get_pdf_statistics: Get metadata about the PDF
"""

import fitz  # PyMuPDF library for PDF processing
import os


class PDFExtractor:
    """
    A class to extract text and metadata from PDF files.
    
    Attributes:
        pdf_path (str): Path to the PDF file to be processed
    """
    
    def __init__(self, pdf_path):
        """
        Initialize the PDF extractor with a specific PDF file.
        
        Args:
            pdf_path (str): Full path to the PDF file
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.pdf_path = pdf_path
        self.filename = os.path.basename(pdf_path)
    
    def extract_all_text(self):
        """
        Extract all text from the PDF file, page by page.
        
        Returns:
            tuple: (full_text, num_pages, char_count)
                - full_text (str): Concatenated text from all pages
                - num_pages (int): Total number of pages in the PDF
                - char_count (int): Total characters extracted
                
        Raises:
            Exception: If PDF cannot be opened or processed
        """
        try:
            # Open the PDF file
            pdf_document = fitz.open(self.pdf_path)
            
            # Initialize variables to store results
            full_text = ""
            num_pages = pdf_document.page_count
            
            # Extract text from each page
            for page_num in range(num_pages):
                # Get the page object
                page = pdf_document[page_num]
                
                # Extract text from the page
                page_text = page.get_text()
                
                # Handle empty pages gracefully
                if page_text.strip():  # Only add non-empty pages
                    full_text += page_text + "\n"
            
            # Close the PDF document
            pdf_document.close()
            
            # Calculate total characters
            char_count = len(full_text)
            
            return full_text, num_pages, char_count
            
        except fitz.FileError:
            raise Exception(f"Failed to open PDF: {self.pdf_path}")
        except Exception as e:
            raise Exception(f"Error extracting text from {self.filename}: {str(e)}")
    
    def get_statistics(self):
        """
        Get extraction statistics for the PDF.
        
        Returns:
            dict: Dictionary containing:
                - 'filename': Name of the PDF file
                - 'num_pages': Number of pages
                - 'total_chars': Total characters extracted
                - 'avg_chars_per_page': Average characters per page
        """
        full_text, num_pages, char_count = self.extract_all_text()
        
        # Calculate average characters per page
        avg_chars_per_page = char_count // num_pages if num_pages > 0 else 0
        
        statistics = {
            'filename': self.filename,
            'num_pages': num_pages,
            'total_chars': char_count,
            'avg_chars_per_page': avg_chars_per_page
        }
        
        return statistics, full_text


def get_all_pdfs_from_folder(folder_path):
    """
    Get all PDF files from a specific folder.
    
    Args:
        folder_path (str): Path to the folder containing PDFs
        
    Returns:
        list: List of PDF file paths found in the folder
        
    Raises:
        FileNotFoundError: If folder doesn't exist
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Find all PDF files in the folder
    pdf_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith('.pdf'):
            full_path = os.path.join(folder_path, file)
            pdf_files.append(full_path)
    
    return sorted(pdf_files)  # Sort alphabetically for consistent processing
