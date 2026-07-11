"""
EXAMPLE USAGE AND CODE DEMONSTRATIONS
======================================

This file demonstrates how to use the Smart Legal Assistance modules.
Copy and adapt these examples for your needs.
"""

# ==============================================================================
# EXAMPLE 1: BASIC PDF EXTRACTION
# ==============================================================================

def example_basic_extraction():
    """
    Extract text from a single PDF file.
    """
    from src.pdf_extractor import PDFExtractor
    
    # Create an extractor for a specific PDF
    extractor = PDFExtractor("legal_pdfs/contract.pdf")
    
    # Extract all text
    full_text, num_pages, char_count = extractor.extract_all_text()
    
    # Display results
    print(f"File: contract.pdf")
    print(f"Pages: {num_pages}")
    print(f"Characters: {char_count}")
    print(f"First 500 characters:\n{full_text[:500]}\n")


# ==============================================================================
# EXAMPLE 2: GET EXTRACTION STATISTICS
# ==============================================================================

def example_get_statistics():
    """
    Get detailed statistics about a PDF file.
    """
    from src.pdf_extractor import PDFExtractor
    
    extractor = PDFExtractor("legal_pdfs/agreement.pdf")
    stats, full_text = extractor.get_statistics()
    
    # Display statistics
    print("PDF Statistics:")
    print(f"  Filename: {stats['filename']}")
    print(f"  Pages: {stats['num_pages']}")
    print(f"  Total Characters: {stats['total_chars']}")
    print(f"  Avg Characters/Page: {stats['avg_chars_per_page']}\n")


# ==============================================================================
# EXAMPLE 3: FIND ALL PDFS IN FOLDER
# ==============================================================================

def example_find_all_pdfs():
    """
    Find and list all PDF files in a folder.
    """
    from src.pdf_extractor import get_all_pdfs_from_folder
    
    # Get all PDFs
    pdf_files = get_all_pdfs_from_folder("legal_pdfs")
    
    print(f"Found {len(pdf_files)} PDF files:")
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"  {idx}. {pdf_path}")
    print()


# ==============================================================================
# EXAMPLE 4: CLEAN TEXT PREPROCESSING
# ==============================================================================

def example_text_cleaning():
    """
    Clean and normalize text.
    """
    from src.preprocessing import Preprocessor
    
    preprocessor = Preprocessor()
    
    # Example raw text
    raw_text = """
    LEGAL AGREEMENT: This  CONTRACT---is a  legal   document!!!
    Page 1
    The parties   agree   to   the   following   terms   &   conditions.
    """
    
    # Clean the text
    cleaned_text = preprocessor.clean_text(raw_text)
    
    print("Original Text:")
    print(raw_text)
    print("\nCleaned Text:")
    print(cleaned_text)
    print()


# ==============================================================================
# EXAMPLE 5: TEXT TOKENIZATION
# ==============================================================================

def example_tokenization():
    """
    Split text into individual words (tokens).
    """
    from src.preprocessing import Preprocessor
    
    preprocessor = Preprocessor()
    
    text = "this is a legal agreement between parties and companies"
    
    # Tokenize
    tokens = preprocessor.tokenize_text(text)
    
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Token Count: {len(tokens)}\n")


# ==============================================================================
# EXAMPLE 6: COMPLETE PREPROCESSING PIPELINE
# ==============================================================================

def example_full_preprocessing():
    """
    Run the complete preprocessing pipeline (clean + tokenize + stats).
    """
    from src.preprocessing import Preprocessor
    
    preprocessor = Preprocessor()
    
    raw_text = """
    CONFIDENTIAL AGREEMENT
    
    This agreement is entered into between Party A and Party B.
    The parties agree to maintain confidentiality.
    
    Page 1 of 5
    """
    
    # Run full preprocessing
    result = preprocessor.preprocess_full_text(raw_text)
    
    print("Preprocessing Results:")
    print(f"  Cleaned Text: {result['cleaned_text']}")
    print(f"  Tokens: {result['tokens']}")
    print(f"  Token Count: {result['token_count']}")
    print(f"  Character Count: {result['char_count']}\n")


# ==============================================================================
# EXAMPLE 7: BATCH PROCESS MULTIPLE PDFS
# ==============================================================================

def example_batch_processing():
    """
    Process multiple PDF files in batch.
    """
    from src.pdf_extractor import PDFExtractor, get_all_pdfs_from_folder
    from src.preprocessing import Preprocessor
    import json
    import os
    
    # Setup
    pdf_folder = "legal_pdfs"
    output_folder = "processed_data"
    preprocessor = Preprocessor()
    
    # Create output folder if needed
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all PDFs
    pdf_files = get_all_pdfs_from_folder(pdf_folder)
    
    print(f"Processing {len(pdf_files)} PDF files...\n")
    
    # Process each PDF
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"Processing: {filename}")
        
        # Extract
        extractor = PDFExtractor(pdf_path)
        full_text, num_pages, char_count = extractor.extract_all_text()
        
        # Preprocess
        result = preprocessor.preprocess_full_text(full_text)
        
        # Prepare output data
        output_data = {
            "file_name": filename,
            "pages": num_pages,
            "total_chars_extracted": char_count,
            "total_words": result['token_count'],
            "clean_text": result['cleaned_text'][:200] + "...",
            "full_clean_text_length": len(result['cleaned_text'])
        }
        
        # Save as JSON
        json_filename = filename.replace('.pdf', '_processed.json')
        json_path = os.path.join(output_folder, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        
        print(f"  ✓ Saved: {json_filename}")
        print(f"  ✓ Pages: {num_pages}, Words: {result['token_count']}\n")


# ==============================================================================
# EXAMPLE 8: CALCULATE STATISTICS
# ==============================================================================

def example_statistics():
    """
    Calculate and display statistics from processed PDFs.
    """
    from src.pdf_extractor import PDFExtractor, get_all_pdfs_from_folder
    from src.preprocessing import Preprocessor
    
    pdf_folder = "legal_pdfs"
    preprocessor = Preprocessor()
    
    # Get all PDFs
    pdf_files = get_all_pdfs_from_folder(pdf_folder)
    
    if not pdf_files:
        print("No PDF files found!")
        return
    
    # Initialize statistics
    total_pages = 0
    total_chars = 0
    total_words = 0
    file_count = 0
    
    print("Processing statistics:\n")
    
    # Process each PDF
    for pdf_path in pdf_files:
        extractor = PDFExtractor(pdf_path)
        text, pages, chars = extractor.extract_all_text()
        result = preprocessor.preprocess_full_text(text)
        
        # Accumulate statistics
        total_pages += pages
        total_chars += chars
        total_words += result['token_count']
        file_count += 1
        
        print(f"  {os.path.basename(pdf_path)}: {pages} pages, {result['token_count']} words")
    
    # Display summary
    print(f"\nSummary Statistics:")
    print(f"  Total files: {file_count}")
    print(f"  Total pages: {total_pages}")
    print(f"  Total characters: {total_chars}")
    print(f"  Total words: {total_words}")
    
    if file_count > 0:
        print(f"  Average words per file: {total_words // file_count}")
    print()


# ==============================================================================
# EXAMPLE 9: ERROR HANDLING
# ==============================================================================

def example_error_handling():
    """
    Handle errors gracefully when processing PDFs.
    """
    from src.pdf_extractor import PDFExtractor
    
    pdf_files = [
        "legal_pdfs/exists.pdf",
        "legal_pdfs/does_not_exist.pdf",
        "legal_pdfs/corrupted.pdf"
    ]
    
    for pdf_path in pdf_files:
        try:
            extractor = PDFExtractor(pdf_path)
            text, pages, chars = extractor.extract_all_text()
            print(f"✓ Successfully processed: {pdf_path}")
        except FileNotFoundError:
            print(f"✗ File not found: {pdf_path}")
        except Exception as e:
            print(f"✗ Error processing {pdf_path}: {str(e)}")


# ==============================================================================
# EXAMPLE 10: CUSTOM TEXT ANALYSIS
# ==============================================================================

def example_custom_analysis():
    """
    Perform custom analysis on processed text.
    """
    from src.pdf_extractor import PDFExtractor
    from src.preprocessing import Preprocessor
    
    extractor = PDFExtractor("legal_pdfs/contract.pdf")
    text, _, _ = extractor.extract_all_text()
    
    preprocessor = Preprocessor()
    result = preprocessor.preprocess_full_text(text)
    
    tokens = result['tokens']
    
    print("Text Analysis:")
    print(f"  Total tokens: {len(tokens)}")
    print(f"  Unique tokens: {len(set(tokens))}")
    print(f"  Vocabulary: {len(set(tokens)) / len(tokens) * 100:.1f}%")
    
    # Find most common words
    from collections import Counter
    word_counts = Counter(tokens)
    print(f"\n  Top 10 most common words:")
    for word, count in word_counts.most_common(10):
        print(f"    - {word}: {count}")
    print()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    """
    Run examples. Uncomment the example you want to test.
    """
    import os
    
    # Example 1: Basic extraction
    print("=" * 70)
    print("EXAMPLE 1: Basic PDF Extraction")
    print("=" * 70)
    if os.path.exists("legal_pdfs/contract.pdf"):
        example_basic_extraction()
    else:
        print("Placeholder PDF not found. Add real PDFs to legal_pdfs/ folder\n")
    
    # Example 2: Find all PDFs
    print("=" * 70)
    print("EXAMPLE 2: Find All PDFs")
    print("=" * 70)
    example_find_all_pdfs()
    
    # Example 3: Text cleaning
    print("=" * 70)
    print("EXAMPLE 3: Text Cleaning")
    print("=" * 70)
    example_text_cleaning()
    
    # Example 4: Tokenization
    print("=" * 70)
    print("EXAMPLE 4: Tokenization")
    print("=" * 70)
    example_tokenization()
    
    # Example 5: Full preprocessing
    print("=" * 70)
    print("EXAMPLE 5: Full Preprocessing Pipeline")
    print("=" * 70)
    example_full_preprocessing()
    
    print("\nTo use other examples, uncomment them in the __main__ block")
    print("or call the functions directly in your code.\n")
