"""
Text Preprocessing Module
=========================
Handles cleaning and preprocessing of extracted legal text.

Functions:
    - clean_text: Remove special characters and normalize text
    - tokenize_text: Split text into individual words
    - preprocess_full_text: Complete preprocessing pipeline
    - TextPreprocessor: Advanced preprocessing with lemmatization
"""

import re  # Regular expressions for text cleaning
import string  # Built-in string constants
import spacy  # Advanced NLP processing with lemmatization
from typing import List, Dict


class Preprocessor:
    """
    A class to preprocess extracted text from legal documents.
    
    This class handles:
    - Converting text to lowercase
    - Removing extra spaces
    - Removing special characters
    - Removing page numbers
    - Tokenizing text
    """
    
    # Legal keywords to preserve (examples)
    LEGAL_KEYWORDS = {
        'agreement', 'contract', 'liability', 'indemnify', 'warrant',
        'represent', 'consideration', 'severability', 'governing',
        'jurisdiction', 'amendment', 'waiver', 'obligation', 'confidential',
        'intellectual property', 'trademark', 'copyright', 'patent',
        'breach', 'remedy', 'damages', 'plaintiff', 'defendant'
    }
    
    def __init__(self):
        """Initialize the preprocessor with default settings."""
        self.processed_count = 0
    
    def convert_to_lowercase(self, text):
        """
        Convert all text to lowercase.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Lowercase text
        """
        return text.lower()
    
    def remove_extra_spaces(self, text):
        """
        Remove extra spaces and normalize whitespace.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized spaces
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def remove_page_numbers(self, text):
        """
        Remove page numbers and common page markers.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text without page numbers
        """
        # Remove patterns like "Page 1", "p. 1", "- 1 -"
        text = re.sub(r'page\s+\d+', '', text)
        text = re.sub(r'p\.\s+\d+', '', text)
        text = re.sub(r'-\s+\d+\s+-', '', text)
        text = re.sub(r'\|\s+\d+\s+\|', '', text)
        return text
    
    def remove_special_characters(self, text):
        """
        Remove unnecessary special characters while keeping punctuation.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with cleaned special characters
        """
        # Keep alphanumeric, spaces, and basic punctuation (., -, (), etc.)
        # This preserves legal terms like "LLC", "Inc.", etc.
        text = re.sub(r'[^\w\s\.\-\(\)\,\;\:\']', '', text)
        return text
    
    def remove_unnecessary_symbols(self, text):
        """
        Remove unnecessary symbols and formatting characters.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with removed symbols
        """
        # Remove bullet points, stars, dashes used for formatting
        text = re.sub(r'[\•\★\─\═]', '', text)
        # Remove multiple punctuation marks
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'-{2,}', '-', text)
        return text
    
    def tokenize_text(self, text):
        """
        Split text into individual words (tokens).
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of individual word tokens
        """
        # Split by spaces and punctuation
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def clean_text(self, text):
        """
        Complete text cleaning pipeline.
        
        This method applies all cleaning steps in sequence:
        1. Convert to lowercase
        2. Remove page numbers
        3. Remove special characters
        4. Remove unnecessary symbols
        5. Remove extra spaces
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Cleaned text ready for analysis
        """
        # Step 1: Convert to lowercase
        text = self.convert_to_lowercase(text)
        
        # Step 2: Remove page numbers
        text = self.remove_page_numbers(text)
        
        # Step 3: Remove special characters
        text = self.remove_special_characters(text)
        
        # Step 4: Remove unnecessary symbols
        text = self.remove_unnecessary_symbols(text)
        
        # Step 5: Remove extra spaces
        text = self.remove_extra_spaces(text)
        
        return text
    
    def preprocess_full_text(self, text):
        """
        Full preprocessing pipeline including cleaning and tokenization.
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            dict: Dictionary containing:
                - 'cleaned_text': Fully cleaned text
                - 'tokens': List of word tokens
                - 'token_count': Number of tokens (words)
                - 'char_count': Character count after cleaning
        """
        # Apply cleaning
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize_text(cleaned_text)
        
        # Calculate statistics
        token_count = len(tokens)
        char_count = len(cleaned_text)
        
        # Increment processing counter
        self.processed_count += 1
        
        # Return complete preprocessing result
        result = {
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'token_count': token_count,
            'char_count': char_count
        }
        
        return result


class TextPreprocessor:
    """
    Advanced text preprocessor with lemmatization and legal-specific enhancements.
    
    Features:
    - Spacy-based lemmatization
    - Legal citation removal (e.g., "Smith v. Jones")
    - Legal domain stopword handling
    - Sentence tokenization
    - Enhanced NLP pipeline
    """
    
    def __init__(self, model: str = "en_core_web_sm"):
        """
        Initialize TextPreprocessor with spacy model.
        
        Args:
            model (str): Spacy model name (default: en_core_web_sm)
        """
        try:
            self.nlp = spacy.load(model)
            print(f"✓ Loaded spacy model: {model}")
        except OSError:
            print(f"⚠️  Spacy model '{model}' not found. Installing...")
            import subprocess
            subprocess.run([f"python -m spacy download {model}"], shell=True)
            self.nlp = spacy.load(model)
        
        # Legal domain stopwords to remove
        self.legal_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'which', 'who', 'whom', 'where', 'when', 'why', 'how'
        }
        
        # Legal keywords to preserve (always keep these)
        self.legal_keywords = {
            'agree', 'contract', 'liability', 'indemnify', 'warrant',
            'represent', 'consideration', 'severability', 'govern',
            'jurisdiction', 'amend', 'waive', 'obligate', 'confidential',
            'intellectual', 'property', 'trademark', 'copyright', 'patent',
            'breach', 'remedy', 'damages', 'plaintiff', 'defendant', 'party',
            'sign', 'execute', 'deliver', 'enforce', 'terminate', 'renew'
        }
    
    def _remove_legal_citations(self, text: str) -> str:
        """
        Remove legal case citations (e.g., "Smith v. Jones", "123 U.S. 456").
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text without citations
        """
        # Remove case names (e.g., "Smith v. Jones", "ABC Inc. v. XYZ Corp.")
        text = re.sub(r'\b[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+', '', text)
        
        # Remove U.S. reporter citations (e.g., "123 U.S. 456")
        text = re.sub(r'\d+\s+U\.S\.\s+\d+', '', text)
        text = re.sub(r'\d+\s+F\.\d*d\s+\d+', '', text)  # Federal Reporter
        text = re.sub(r'\d+\s+N\.E\.\d*d\s+\d+', '', text)  # North Eastern Reporter
        text = re.sub(r'\d+\s+S\.E\.\d*d\s+\d+', '', text)  # South Eastern Reporter
        
        return text
    
    def _remove_citations_and_refs(self, text: str) -> str:
        """
        Remove footnotes, citations, and reference markers.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Remove footnote numbers like [1], [123]
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove endnote references
        text = re.sub(r'\(note \d+\)', '', text, flags=re.IGNORECASE)
        
        # Remove URL citations
        text = re.sub(r'http[s]?://\S+', '', text)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text with domain-aware enhancements.
        
        Pipeline:
        1. Lowercase conversion
        2. Remove legal citations
        3. Remove citations and references
        4. Remove special characters (preserve legal terms)
        5. Normalize whitespace
        6. Lemmatization
        7. Remove stopwords (except legal keywords)
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned and lemmatized text
        """
        # Step 1: Lowercase
        text = text.lower()
        
        # Step 2: Remove legal citations
        text = self._remove_legal_citations(text)
        
        # Step 3: Remove citations and references
        text = self._remove_citations_and_refs(text)
        
        # Step 4: Remove special characters (keep alphanumeric, punctuation)
        text = re.sub(r'[^\w\s\.\-\(\)\,\;\:\']', '', text)
        
        # Step 5: Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Step 6-7: Process with spacy for lemmatization and stopword removal
        doc = self.nlp(text)
        
        lemmatized_tokens = []
        for token in doc:
            # Skip if it's a stopword (unless it's a legal keyword)
            if token.text in self.legal_stopwords and token.text not in self.legal_keywords:
                continue
            
            # Skip if it's punctuation or very short
            if token.is_punct or len(token.text) < 2:
                continue
            
            # Use lemma if available, otherwise use text
            lemma = token.lemma_ if token.lemma_ != "-PRON-" else token.text
            lemmatized_tokens.append(lemma)
        
        # Rejoin tokens
        cleaned_text = " ".join(lemmatized_tokens)
        
        return cleaned_text
    
    def get_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using spacy.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of sentences
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences
    
    def get_tokens(self, text: str) -> List[str]:
        """
        Get lemmatized tokens from text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of lemmatized tokens
        """
        doc = self.nlp(text)
        tokens = [token.lemma_ if token.lemma_ != "-PRON-" else token.text 
                 for token in doc if not token.is_stop and not token.is_punct]
        return tokens
    
    def get_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict: Named entities by type (PERSON, ORG, GPE, DATE, etc.)
        """
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            ent_type = ent.label_
            if ent_type not in entities:
                entities[ent_type] = []
            entities[ent_type].append(ent.text)
        
        return entities
    
    def get_statistics(self, text: str) -> Dict:
        """
        Get preprocessing statistics.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict: Statistics about the text
        """
        doc = self.nlp(text)
        sentences = [sent for sent in doc.sents]
        tokens = [token for token in doc if not token.is_punct]
        
        return {
            "char_count": len(text),
            "sentence_count": len(sentences),
            "token_count": len(tokens),
            "avg_sentence_length": len(tokens) / max(len(sentences), 1),
            "unique_tokens": len(set(token.text for token in tokens))
        }
