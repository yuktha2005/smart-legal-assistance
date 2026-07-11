"""
Storage Module
==============
Persistent storage for unsupervised learning results.

Saves:
- Processed chunks (JSON)
- Embeddings (NPY binary format)
- Cluster results (JSON)
- Topic results (JSON)
- Knowledge discoveries (JSON)

Functions:
    - StorageManager: Handle all persistence operations
"""

import os
import json
import numpy as np
from typing import Dict, List, Any
from datetime import datetime


class StorageManager:
    """
    Manage persistent storage of unsupervised learning results.
    
    Stores data in structured format for easy retrieval and analysis.
    Maintains metadata for reproducibility.
    """
    
    def __init__(self, output_folder: str = "processed_data"):
        """
        Initialize storage manager.
        
        Args:
            output_folder (str): Directory to store all results
        """
        self.output_folder = output_folder
        self.metadata = {
            "created": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        # Create output folder if needed
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"📁 Created output folder: {output_folder}")
        else:
            print(f"📁 Using output folder: {output_folder}")
    
    def save_chunks(self, chunks: List[str], 
                   metadata_list: List[Dict] = None,
                   filename: str = "processed_chunks.json") -> str:
        """
        Save document chunks to JSON.
        
        Args:
            chunks (List[str]): Text chunks
            metadata_list (List[Dict], optional): Metadata for each chunk
            filename (str): Output filename
            
        Returns:
            str: Path to saved file
        """
        filepath = os.path.join(self.output_folder, filename)
        
        data = {
            "metadata": self.metadata,
            "n_chunks": len(chunks),
            "chunks": [
                {
                    "index": i,
                    "text": chunk,
                    "length": len(chunk),
                    "extra_metadata": metadata_list[i] if metadata_list else None
                }
                for i, chunk in enumerate(chunks)
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Chunks saved: {filename}")
        return filepath
    
    def save_embeddings(self, embeddings: np.ndarray,
                       filename: str = "embeddings.npy") -> str:
        """
        Save embeddings to binary NPY format.
        
        Args:
            embeddings (np.ndarray): Embedding vectors
            filename (str): Output filename
            
        Returns:
            str: Path to saved file
        """
        filepath = os.path.join(self.output_folder, filename)
        np.save(filepath, embeddings)
        
        size_mb = embeddings.nbytes / 1024 / 1024
        print(f"💾 Embeddings saved: {filename} ({size_mb:.2f} MB)")
        
        return filepath
    
    def load_embeddings(self, filename: str = "embeddings.npy") -> np.ndarray:
        """
        Load embeddings from NPY file.
        
        Args:
            filename (str): Filename to load
            
        Returns:
            np.ndarray: Loaded embeddings
        """
        filepath = os.path.join(self.output_folder, filename)
        embeddings = np.load(filepath)
        print(f"📂 Embeddings loaded: {filename} (shape: {embeddings.shape})")
        return embeddings
    
    def save_clustering_results(self, results: Dict,
                               filename: str = "cluster_results.json") -> str:
        """
        Save clustering results to JSON.
        
        Args:
            results (Dict): Clustering results
            filename (str): Output filename
            
        Returns:
            str: Path to saved file
        """
        filepath = os.path.join(self.output_folder, filename)
        
        data = {
            "metadata": self.metadata,
            "results": results
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Clustering results saved: {filename}")
        return filepath
    
    def save_topic_results(self, results: Dict,
                          filename: str = "topic_results.json") -> str:
        """
        Save topic modeling results to JSON.
        
        Args:
            results (Dict): Topic results
            filename (str): Output filename
            
        Returns:
            str: Path to saved file
        """
        filepath = os.path.join(self.output_folder, filename)
        
        data = {
            "metadata": self.metadata,
            "results": results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Topic results saved: {filename}")
        return filepath
    
    def save_knowledge_discovery(self, results: Dict,
                                filename: str = "knowledge_discovery.json") -> str:
        """
        Save knowledge discovery results to JSON.
        
        Args:
            results (Dict): Discovery results
            filename (str): Output filename
            
        Returns:
            str: Path to saved file
        """
        filepath = os.path.join(self.output_folder, filename)
        
        data = {
            "metadata": self.metadata,
            "results": results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Knowledge discovery saved: {filename}")
        return filepath
    
    def save_summary(self, summary: Dict,
                    filename: str = "unsupervised_summary.json") -> str:
        """
        Save comprehensive summary of all analyses.
        
        Args:
            summary (Dict): Complete summary
            filename (str): Output filename
            
        Returns:
            str: Path to saved file
        """
        filepath = os.path.join(self.output_folder, filename)
        
        data = {
            "metadata": self.metadata,
            "summary": summary
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Summary saved: {filename}")
        return filepath
    
    def load_json(self, filename: str) -> Dict:
        """
        Load any JSON file.
        
        Args:
            filename (str): Filename to load
            
        Returns:
            Dict: Loaded JSON data
        """
        filepath = os.path.join(self.output_folder, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📂 Loaded: {filename}")
        return data
    
    def list_files(self) -> List[str]:
        """
        List all files in output folder.
        
        Returns:
            List[str]: Filenames
        """
        files = os.listdir(self.output_folder)
        print(f"\n📂 Files in {self.output_folder}:")
        for f in sorted(files):
            filepath = os.path.join(self.output_folder, f)
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"   • {f} ({size:.1f} KB)")
        return files
    
    def get_file_path(self, filename: str) -> str:
        """
        Get full path to a file.
        
        Args:
            filename (str): Filename
            
        Returns:
            str: Full path
        """
        return os.path.join(self.output_folder, filename)
