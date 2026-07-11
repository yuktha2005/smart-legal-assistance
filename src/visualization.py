"""
Visualization Module
====================
Create publication-quality visualizations for unsupervised learning results.

Generates:
- Cluster visualizations (2D PCA, 3D scatter)
- Topic distributions (bar charts, pie charts)
- Elbow curves and silhouette analysis
- Word clouds for clusters and topics
- Heatmaps for cluster-topic relationships

Functions:
    - VisualizationManager: Handle all plotting
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class VisualizationManager:
    """
    Create professional visualizations for unsupervised learning results.
    """
    
    def __init__(self, output_folder: str = "processed_data"):
        """
        Initialize visualization manager.
        
        Args:
            output_folder (str): Directory to save visualizations
        """
        self.output_folder = output_folder
    
    def plot_cluster_2d(self, embeddings: np.ndarray,
                       labels: np.ndarray,
                       title: str = "Document Clusters (2D PCA Projection)",
                       output_path: str = None) -> str:
        """
        Visualize clusters in 2D using PCA.
        
        Args:
            embeddings (np.ndarray): Document embeddings
            labels (np.ndarray): Cluster labels
            title (str): Plot title
            output_path (str, optional): Save path
            
        Returns:
            str: Path to saved figure
        """
        from sklearn.decomposition import PCA
        
        print("📈 Creating 2D cluster visualization...")
        
        # Reduce to 2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot each cluster
        n_clusters = len(np.unique(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[colors[cluster_id]], 
                      label=f'Cluster {cluster_id}',
                      alpha=0.6, s=100, edgecolors='k', linewidth=0.5)
        
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        if output_path is None:
            output_path = f"{self.output_folder}/cluster_visualization_2d.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_path}")
        plt.close()
        
        return output_path
    
    def plot_topic_distribution(self, topic_info: Dict,
                               output_path: str = None) -> str:
        """
        Visualize topic distribution.
        
        Args:
            topic_info (Dict): Topic information
            output_path (str, optional): Save path
            
        Returns:
            str: Path to saved figure
        """
        print("📊 Creating topic distribution visualization...")
        
        # Extract data
        topics = list(topic_info.keys())
        sizes = [topic_info[t]['document_count'] for t in topics]
        labels = [topic_info[t]['label'] for t in topics]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(topics)))
        ax1.barh([f"Topic {t}\n{labels[i]}" for i, t in enumerate(topics)], 
                sizes, color=colors)
        ax1.set_xlabel('Number of Documents', fontsize=11)
        ax1.set_title('Topic Distribution (Document Count)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Pie chart
        ax2.pie(sizes, labels=[f"T{t}" for t in topics], autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax2.set_title('Topic Distribution (Percentage)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        if output_path is None:
            output_path = f"{self.output_folder}/topic_distribution.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_path}")
        plt.close()
        
        return output_path
    
    def plot_elbow_curve(self, inertias: List[float],
                        silhouette_scores: List[float],
                        optimal_k: int = None,
                        output_path: str = None) -> str:
        """
        Visualize Elbow Method and Silhouette scores.
        
        Args:
            inertias (List[float]): Inertia values
            silhouette_scores (List[float]): Silhouette scores
            optimal_k (int, optional): Optimal K value
            output_path (str, optional): Save path
            
        Returns:
            str: Path to saved figure
        """
        print("📉 Creating elbow curve visualization...")
        
        k_values = range(2, 2 + len(inertias))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow curve
        ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
        ax1.set_ylabel('Inertia', fontsize=11)
        ax1.set_title('Elbow Method', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette scores
        ax2.plot(k_values, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        if optimal_k is not None:
            ax2.axvline(x=optimal_k, color='g', linestyle='--', linewidth=2,
                       label=f'Optimal k={optimal_k}')
            ax2.legend(fontsize=10)
        ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
        ax2.set_ylabel('Silhouette Score', fontsize=11)
        ax2.set_title('Silhouette Analysis', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        if output_path is None:
            output_path = f"{self.output_folder}/elbow_curve.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_path}")
        plt.close()
        
        return output_path
    
    def plot_cluster_topic_heatmap(self, cluster_topic_rel: Dict,
                                  output_path: str = None) -> str:
        """
        Create heatmap of cluster-topic relationships.
        
        Args:
            cluster_topic_rel (Dict): Cluster-topic relationship data
            output_path (str, optional): Save path
            
        Returns:
            str: Path to saved figure
        """
        print("🔥 Creating cluster-topic heatmap...")
        
        # Build matrix
        clusters = sorted(cluster_topic_rel.keys())
        topics = set()
        for cluster_data in cluster_topic_rel.values():
            topics.update(cluster_data.keys())
        topics = sorted(topics)
        
        # Create matrix
        matrix = np.zeros((len(clusters), len(topics)))
        for i, cluster_id in enumerate(clusters):
            for j, topic_id in enumerate(topics):
                if topic_id in cluster_topic_rel[cluster_id]:
                    matrix[i, j] = cluster_topic_rel[cluster_id][topic_id]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, len(clusters) * 0.8))
        
        sns.heatmap(matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                   xticklabels=[f"T{t}" for t in topics],
                   yticklabels=[f"C{c}" for c in clusters],
                   cbar_kws={'label': 'Document Count'},
                   ax=ax)
        
        ax.set_xlabel('Topics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Clusters', fontsize=12, fontweight='bold')
        ax.set_title('Cluster-Topic Relationship', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        if output_path is None:
            output_path = f"{self.output_folder}/cluster_topic_heatmap.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_path}")
        plt.close()
        
        return output_path
    
    def plot_cluster_sizes(self, cluster_labels: np.ndarray,
                          output_path: str = None) -> str:
        """
        Visualize cluster size distribution.
        
        Args:
            cluster_labels (np.ndarray): Cluster assignments
            output_path (str, optional): Save path
            
        Returns:
            str: Path to saved figure
        """
        print("📊 Creating cluster size visualization...")
        
        # Count documents per cluster
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique_clusters)))
        ax.bar([f"Cluster {c}" for c in unique_clusters], counts, color=colors, edgecolor='black')
        
        ax.set_ylabel('Number of Documents', fontsize=12)
        ax.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(counts):
            ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        if output_path is None:
            output_path = f"{self.output_folder}/cluster_sizes.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_path}")
        plt.close()
        
        return output_path
    
    def create_summary_dashboard(self, embeddings: np.ndarray,
                                cluster_labels: np.ndarray,
                                topic_info: Dict,
                                cluster_topic_rel: Dict,
                                output_path: str = None) -> str:
        """
        Create a comprehensive summary dashboard.
        
        Args:
            embeddings (np.ndarray): Document embeddings
            cluster_labels (np.ndarray): Cluster assignments
            topic_info (Dict): Topic information
            cluster_topic_rel (Dict): Cluster-topic relationships
            output_path (str, optional): Save path
            
        Returns:
            str: Path to saved figure
        """
        print("📋 Creating summary dashboard...")
        
        from sklearn.decomposition import PCA
        
        # Create 2x2 grid
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Clusters (2D)
        ax1 = fig.add_subplot(gs[0, 0])
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        n_clusters = len(np.unique(cluster_labels))
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=[colors[cluster_id]], label=f'C{cluster_id}',
                       alpha=0.6, s=50)
        ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax1.set_title("Clusters (2D PCA)", fontweight='bold')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Cluster sizes
        ax2 = fig.add_subplot(gs[0, 1])
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        ax2.barh([f"C{c}" for c in unique_clusters], counts, color=colors)
        ax2.set_xlabel('Count')
        ax2.set_title('Cluster Sizes', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Topic distribution
        ax3 = fig.add_subplot(gs[1, 0])
        topics = list(topic_info.keys())
        sizes = [topic_info[t]['document_count'] for t in topics]
        topic_colors = plt.cm.Pastel1(np.linspace(0, 1, len(topics)))
        ax3.bar([f"T{t}" for t in topics], sizes, color=topic_colors, edgecolor='black')
        ax3.set_ylabel('Count')
        ax3.set_title('Topic Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Cluster-topic heatmap (simplified)
        ax4 = fig.add_subplot(gs[1, 1])
        clusters = sorted(cluster_topic_rel.keys())
        topics_list = sorted(set(t for d in cluster_topic_rel.values() for t in d.keys()))
        
        matrix = np.zeros((len(clusters), len(topics_list)))
        for i, c in enumerate(clusters):
            for j, t in enumerate(topics_list):
                if t in cluster_topic_rel[c]:
                    matrix[i, j] = cluster_topic_rel[c][t]
        
        im = ax4.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax4.set_xticks(range(len(topics_list)))
        ax4.set_yticks(range(len(clusters)))
        ax4.set_xticklabels([f"T{t}" for t in topics_list], fontsize=9)
        ax4.set_yticklabels([f"C{c}" for c in clusters], fontsize=9)
        ax4.set_title('Cluster-Topic Matrix', fontweight='bold')
        plt.colorbar(im, ax=ax4, label='Count')
        
        fig.suptitle('Unsupervised Learning Summary Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save figure
        if output_path is None:
            output_path = f"{self.output_folder}/summary_dashboard.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_path}")
        plt.close()
        
        return output_path
