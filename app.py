"""
Smart Legal Assistance - Phase 1
================================
Main application for processing legal PDF documents.

This script:
1. Reads all PDF files from legal_pdfs/ folder
2. Extracts text using PyMuPDF
3. Preprocesses text (cleaning, tokenization)
4. Saves processed data as TXT and JSON
5. Displays comprehensive statistics

Author: Senior NLP Engineer
Date: 2026
"""

import os
import json
from src.pdf_extractor import PDFExtractor, get_all_pdfs_from_folder
from src.preprocessing import Preprocessor


def create_output_directory(output_path):
    """
    Create the output directory if it doesn't exist.
    
    Args:
        output_path (str): Path to the output directory
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"✓ Created output directory: {output_path}")


def process_single_pdf(pdf_path, preprocessor, output_folder):
    """
    Process a single PDF file: extract, clean, and save.
    
    Args:
        pdf_path (str): Full path to the PDF file
        preprocessor (Preprocessor): Preprocessor instance
        output_folder (str): Path to save processed files
        
    Returns:
        dict: Statistics about the processed PDF
    """
    filename = os.path.basename(pdf_path)
    
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")
    
    try:
        # Step 1: Create PDF extractor
        extractor = PDFExtractor(pdf_path)
        
        # Step 2: Extract text and get statistics
        print("📄 Extracting text from PDF...")
        full_text, num_pages, char_count = extractor.extract_all_text()
        
        # Display extraction statistics
        print(f"   ✓ Pages: {num_pages}")
        print(f"   ✓ Characters extracted: {char_count:,}")
        
        # Step 3: Preprocess the text
        print("🔄 Preprocessing text...")
        preprocessing_result = preprocessor.preprocess_full_text(full_text)
        
        cleaned_text = preprocessing_result['cleaned_text']
        tokens = preprocessing_result['tokens']
        token_count = preprocessing_result['token_count']
        
        print(f"   ✓ Words: {token_count:,}")
        print(f"   ✓ Characters (cleaned): {preprocessing_result['char_count']:,}")
        
        # Step 4: Save extracted text as TXT
        txt_filename = filename.replace('.pdf', '_extracted.txt')
        txt_path = os.path.join(output_folder, txt_filename)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"💾 Saved extracted text: {txt_filename}")
        
        # Step 5: Prepare data for JSON
        output_data = {
            "file_name": filename,
            "pages": num_pages,
            "total_chars_extracted": char_count,
            "total_words": token_count,
            "clean_text": cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text,
            "full_clean_text_length": len(cleaned_text)
        }
        
        # Step 6: Save processed data as JSON
        json_filename = filename.replace('.pdf', '_processed.json')
        json_path = os.path.join(output_folder, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print(f"💾 Saved processed data: {json_filename}")
        
        # Step 7: Return statistics for summary
        return {
            'filename': filename,
            'pages': num_pages,
            'extracted_chars': char_count,
            'words': token_count,
            'status': 'SUCCESS'
        }
    
    except Exception as e:
        print(f"❌ Error processing {filename}: {str(e)}")
        return {
            'filename': filename,
            'status': 'FAILED',
            'error': str(e)
        }


def display_final_statistics(all_stats):
    """
    Display comprehensive statistics about all processed PDFs.
    
    Args:
        all_stats (list): List of statistics dictionaries from processed PDFs
    """
    print(f"\n{'='*60}")
    print("📊 FINAL STATISTICS - PHASE 1 PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    # Filter successful processes
    successful_stats = [s for s in all_stats if s['status'] == 'SUCCESS']
    failed_count = len(all_stats) - len(successful_stats)
    
    print(f"\n✓ Total PDFs processed successfully: {len(successful_stats)}")
    if failed_count > 0:
        print(f"❌ Failed PDFs: {failed_count}")
    
    # Calculate aggregate statistics
    if successful_stats:
        total_pages = sum(s['pages'] for s in successful_stats)
        total_chars = sum(s['extracted_chars'] for s in successful_stats)
        total_words = sum(s['words'] for s in successful_stats)
        
        print(f"\n📈 Aggregate Metrics:")
        print(f"   • Total pages processed: {total_pages:,}")
        print(f"   • Total characters extracted: {total_chars:,}")
        print(f"   • Total words extracted: {total_words:,}")
        
        if len(successful_stats) > 0:
            avg_words_per_doc = total_words // len(successful_stats)
            print(f"   • Average words per document: {avg_words_per_doc:,}")
        
        print(f"\n📄 Individual File Summary:")
        for stat in successful_stats:
            print(f"   • {stat['filename']}: {stat['pages']} pages, {stat['words']:,} words")


def main():
    """
    Main function to orchestrate the entire Phase 1 processing pipeline.
    """
    print("\n" + "="*60)
    print("🚀 SMART LEGAL ASSISTANCE - PHASE 1")
    print("="*60)
    
    # Configuration
    LEGAL_PDFS_FOLDER = os.path.join(os.path.dirname(__file__), 'legal_pdfs')
    PROCESSED_DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'processed_data')
    
    # Step 1: Create output directory
    print("\n📁 Setting up directories...")
    create_output_directory(PROCESSED_DATA_FOLDER)
    
    # Step 2: Get all PDFs from folder
    print(f"\n🔍 Scanning for PDF files in: {LEGAL_PDFS_FOLDER}")
    try:
        pdf_files = get_all_pdfs_from_folder(LEGAL_PDFS_FOLDER)
        print(f"✓ Found {len(pdf_files)} PDF file(s)")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"Please ensure the 'legal_pdfs' folder exists in: {LEGAL_PDFS_FOLDER}")
        return
    
    # Step 3: Check if PDFs exist
    if len(pdf_files) == 0:
        print("⚠️  No PDF files found in legal_pdfs folder")
        print("Please add PDF files to the legal_pdfs folder and run again")
        return
    
    # Step 4: Initialize preprocessor
    preprocessor = Preprocessor()
    
    # Step 5: Process each PDF
    print(f"\n{'='*60}")
    print("STARTING PDF PROCESSING")
    print(f"{'='*60}")
    
    all_statistics = []
    
    for pdf_path in pdf_files:
        stats = process_single_pdf(pdf_path, preprocessor, PROCESSED_DATA_FOLDER)
        all_statistics.append(stats)
    
    # Step 6: Display final statistics
    display_final_statistics(all_statistics)
    
    print(f"\n{'='*60}")
    print("✅ PHASE 1 PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"\n💾 Processed files saved to: {PROCESSED_DATA_FOLDER}")
    print(f"   - TXT files: Raw extracted text")
    print(f"   - JSON files: Cleaned text and metadata")
    print("\n")


if __name__ == "__main__":
    main()
