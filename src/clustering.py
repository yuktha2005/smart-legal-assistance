"""
Clustering Module
================
Unsupervised clustering of legal documents using K-Means.

Features:
- Automatic optimal cluster count using Elbow Method
- Silhouette Score evaluation
- Cluster statistics and summaries
- Cluster visualization

Functions:
    - ClusterAnalyzer: Main clustering engine
    - determine_optimal_clusters: Find best K
"""

import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt


class ClusterAnalyzer:
    """
    Perform K-Means clustering on legal document embeddings.
    
    Automatically determines optimal number of clusters using:
    - Elbow Method: Looking for "elbow" in inertia curve
    - Silhouette Score: Measuring cluster quality (-1 to 1, higher is better)
    - Davies-Bouldin Index: Cluster separation metric
    """
    
    def __init__(self, embeddings: np.ndarray, n_clusters: Optional[int] = None):
        """
        Initialize the clustering analyzer.
        
        Args:
            embeddings (np.ndarray): Document embeddings (n_docs, n_dims)
            n_clusters (int, optional): Number of clusters. If None, auto-determine.
        """
        self.embeddings = embeddings
        self.n_docs = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]
        self.n_clusters = n_clusters
        self.kmeans = None
        self.labels = None
        self.inertias = None
        self.silhouette_scores = None
        
        print(f"📊 Initialized clustering analyzer")
        print(f"   Documents: {self.n_docs}")
        print(f"   Embedding dimension: {self.embedding_dim}")
    
    def find_optimal_clusters(self, k_range: Tuple[int, int] = (2, 10)) -> int:
        """
        Automatically determine optimal number of clusters.
        
        Uses Elbow Method and Silhouette Score combined.
        
        Args:
            k_range (Tuple[int, int]): Range of K values to test
            
        Returns:
            int: Optimal number of clusters
        """
        print(f"\n🔍 Finding optimal cluster count (k={k_range[0]}-{k_range[1]})...")
        
        self.inertias = []
        self.silhouette_scores = []
        # k must be <= n_samples - 1 for silhouette score to be valid.
        max_k = min(k_range[1], self.n_docs - 1)
        min_k = k_range[0]
        
        if max_k < min_k:
            # Not enough documents to compute silhouette reliably.
            fallback_k = max(2, min(self.n_docs, k_range[0]))
            print(f"⚠️ Not enough samples for silhouette search. Falling back to k={fallback_k}")
            return fallback_k

        k_values = range(min_k, max_k + 1)
        
        # Test different K values
        for k in k_values:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_temp = kmeans_temp.fit_predict(self.embeddings)

            # silhouette requires at least 2 clusters
            if len(set(labels_temp)) < 2:
                continue
            
            inertia = kmeans_temp.inertia_
            silhouette = silhouette_score(self.embeddings, labels_temp)

            
            self.inertias.append(inertia)
            self.silhouette_scores.append(silhouette)
            
            print(f"   k={k}: Inertia={inertia:.2f}, Silhouette={silhouette:.3f}")
        
        # Find optimal K using Silhouette Score (more reliable)
        optimal_idx = np.argmax(self.silhouette_scores)
        optimal_k = list(k_values)[optimal_idx]
        
        print(f"\n✓ Optimal clusters: {optimal_k} (Silhouette: {self.silhouette_scores[optimal_idx]:.3f})")
        
        return optimal_k
    
    def fit(self, n_clusters: Optional[int] = None) -> None:
        """
        Fit K-Means clustering.
        
        Args:
            n_clusters (int, optional): Number of clusters. 
                                       If None, use auto-determined value.
        """
        if n_clusters is None:
            if self.n_clusters is None:
                n_clusters = self.find_optimal_clusters()
            else:
                n_clusters = self.n_clusters
        
        print(f"\n🎯 Fitting K-Means with k={n_clusters}...")
        
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            verbose=1
        )
        
        self.labels = self.kmeans.fit_predict(self.embeddings)
        self.n_clusters = n_clusters
        
        print(f"✓ Clustering complete")
    
    def get_statistics(self) -> Dict:
        """
        Get detailed clustering statistics.
        
        Returns:
            Dict: Statistics about the clusters
        """
        if self.labels is None:
            raise ValueError("Must fit clustering first")
        
        stats = {
            "n_clusters": self.n_clusters,
            "n_documents": self.n_docs,
            "silhouette_score": float(silhouette_score(self.embeddings, self.labels)),
            "inertia": float(self.kmeans.inertia_),
            "clusters": {}
        }
        
        # Per-cluster statistics
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            stats["clusters"][int(cluster_id)] = {
                "size": int(cluster_size),
                "percentage": float(cluster_size / self.n_docs * 100),
                "centroid_norm": float(np.linalg.norm(self.kmeans.cluster_centers_[cluster_id]))
            }
        
        return stats
    
    def get_cluster_assignments(self) -> Dict[int, List[int]]:
        """
        Get document indices for each cluster.
        
        Returns:
            Dict: {cluster_id: [doc_indices]}
        """
        assignments = {}
        for cluster_id in range(self.n_clusters):
            doc_indices = np.where(self.labels == cluster_id)[0].tolist()
            assignments[cluster_id] = doc_indices
        return assignments
    
    def get_closest_documents(self, cluster_id: int, 
                             n: int = 5) -> List[Tuple[int, float]]:
        """
        Get documents closest to cluster centroid.
        
        Args:
            cluster_id (int): Cluster ID
            n (int): Number of documents to return
            
        Returns:
            List[Tuple[int, float]]: [(doc_index, distance), ...]
        """
        cluster_mask = self.labels == cluster_id
        cluster_points = self.embeddings[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        # Distances to centroid
        centroid = self.kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        
        # Get closest N
        closest_idx = np.argsort(distances)[:n]
        results = [(int(cluster_indices[i]), float(distances[i])) 
                   for i in closest_idx]
        
        return results
    
    def visualize_clusters(self, texts: List[str], 
                          output_path: str = "cluster_visualization.png") -> None:
        """
        Visualize clusters using 2D PCA projection.
        
        Args:
            texts (List[str]): Document texts for reference
            output_path (str): Path to save visualization
        """
        from sklearn.decomposition import PCA
        
        print(f"\n📈 Creating cluster visualization...")
        
        # Project to 2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(self.embeddings)
        
        # Create plot
        plt.figure(figsize=(14, 10))
        
        # Plot clusters
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_clusters))
        for cluster_id in range(self.n_clusters):
            mask = self.labels == cluster_id
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                       alpha=0.6, s=100, edgecolors='k', linewidth=0.5)
            
            # Plot centroid
            centroid_2d = pca.transform(self.kmeans.cluster_centers_[cluster_id].reshape(1, -1))
            plt.scatter(centroid_2d[0, 0], centroid_2d[0, 1],
                       c=[colors[cluster_id]], marker='*', s=500,
                       edgecolors='black', linewidth=2)
        
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.title("Legal Document Clusters (PCA Projection)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualization saved to {output_path}")
        plt.close()
    
    def visualize_elbow(self, output_path: str = "elbow_curve.png") -> None:
        """
        Visualize Elbow Method and Silhouette scores.
        
        Args:
            output_path (str): Path to save visualization
        """
        if self.inertias is None:
            print("⚠️  Run find_optimal_clusters first")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        k_values = range(2, 2 + len(self.inertias))
        
        # Elbow curve
        ax1.plot(k_values, self.inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette scores
        ax2.plot(k_values, self.silhouette_scores, 'ro-', linewidth=2, markersize=8)
        optimal_idx = np.argmax(self.silhouette_scores)
        optimal_k = list(k_values)[optimal_idx]
        ax2.axvline(x=optimal_k, color='g', linestyle='--', label=f'Optimal k={optimal_k}')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"💾 Elbow curve saved to {output_path}")
        plt.close()
    
    def to_dict(self) -> Dict:
        """
        Convert clustering results to dictionary format.
        
        Returns:
            Dict: Complete clustering results
        """
        return {
            "n_clusters": int(self.n_clusters),
            "n_documents": int(self.n_docs),
            "statistics": self.get_statistics(),
            "cluster_assignments": {
                str(k): v for k, v in self.get_cluster_assignments().items()
            },
            "labels": self.labels.tolist()
        }
