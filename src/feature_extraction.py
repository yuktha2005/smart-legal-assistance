"""
Feature Extraction Module
=========================
Generates embeddings from legal text using SentenceTransformers.

Uses pre-trained all-MiniLM-L6-v2 model for semantic understanding.
Produces fixed-size dense vectors suitable for clustering and similarity.

Functions:
    - EmbeddingGenerator: Generate embeddings from text chunks
    - batch_embed: Process large documents in batches
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict


class EmbeddingGenerator:
    """
    Generate semantic embeddings for legal text using SentenceTransformers.
    
    The all-MiniLM-L6-v2 model:
    - Produces 384-dimensional embeddings
    - Fine-tuned for semantic similarity
    - Fast inference on CPU
    - Lightweight (~33MB)
    - Suitable for legal document analysis
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator with a pre-trained model.
        
        Args:
            model_name (str): HuggingFace model name
                Default: all-MiniLM-L6-v2 (fast, good quality)
                Alternative: all-mpnet-base-v2 (higher quality, slower)
        """
        print(f"📥 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = 384  # Dimension of all-MiniLM-L6-v2
        print(f"✓ Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text chunk.
        
        Args:
            text (str): Text to embed
            
        Returns:
            np.ndarray: 1D embedding vector (384 dimensions)
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.embedding_dim)
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32,
                   show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple text chunks efficiently.
        
        Args:
            texts (List[str]): List of text chunks
            batch_size (int): Number of texts per batch
            show_progress (bool): Display progress bar
            
        Returns:
            np.ndarray: 2D array of embeddings (n_texts, 384)
        """
        if not texts:
            return np.array([])
        
        print(f"\n📊 Generating embeddings for {len(texts)} chunks...")
        
        # Encode with progress tracking
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        print(f"✓ Generated {len(embeddings)} embeddings")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Memory usage: {embeddings.nbytes / 1024 / 1024:.2f} MB")
        
        return embeddings
    
    def get_similarity(self, embedding1: np.ndarray, 
                      embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1 (np.ndarray): First embedding vector
            embedding2 (np.ndarray): Second embedding vector
            
        Returns:
            float: Similarity score between -1 and 1 (higher = more similar)
        """
        # Normalize embeddings
        norm1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        norm2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(norm1, norm2)
        return float(similarity)
    
    def get_top_similar(self, query_text: str, 
                       chunks: List[str], 
                       embeddings: np.ndarray,
                       top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find top-k most similar chunks to a query.
        
        Args:
            query_text (str): Query text
            chunks (List[str]): List of text chunks
            embeddings (np.ndarray): Pre-computed embeddings
            top_k (int): Number of results to return
            
        Returns:
            List[Tuple[str, float]]: List of (chunk, similarity) pairs
        """
        # Embed query
        query_embedding = self.embed_single(query_text)
        
        # Calculate similarities
        similarities = []
        for i, embedding in enumerate(embeddings):
            sim = self.get_similarity(query_embedding, embedding)
            similarities.append((chunks[i], sim, i))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [(text, sim) for text, sim, _ in similarities[:top_k]]
    
    def save_embeddings(self, embeddings: np.ndarray, 
                       filepath: str) -> None:
        """
        Save embeddings to disk for later use.
        
        Args:
            embeddings (np.ndarray): Embeddings to save
            filepath (str): Path to save file (.npy)
        """
        np.save(filepath, embeddings)
        print(f"💾 Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """
        Load embeddings from disk.
        
        Args:
            filepath (str): Path to embeddings file (.npy)
            
        Returns:
            np.ndarray: Loaded embeddings
        """
        embeddings = np.load(filepath)
        print(f"📂 Embeddings loaded from {filepath}")
        print(f"   Shape: {embeddings.shape}")
        return embeddings


def create_embedding_pipeline(texts: List[str], 
                             model_name: str = "all-MiniLM-L6-v2",
                             batch_size: int = 32) -> Tuple[np.ndarray, EmbeddingGenerator]:
    """
    Create a complete embedding pipeline in one function.
    
    Args:
        texts (List[str]): List of text chunks
        model_name (str): Model to use
        batch_size (int): Batch size for processing
        
    Returns:
        Tuple: (embeddings, generator)
    """
    generator = EmbeddingGenerator(model_name)
    embeddings = generator.embed_batch(texts, batch_size=batch_size)
    return embeddings, generator

