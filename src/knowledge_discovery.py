"""
Knowledge Discovery Module
==========================
Extract actionable insights from clustering and topic modeling results.

Produces:
- Cluster summaries
- Topic summaries
- Cluster-topic relationships
- Pattern insights
- Anomaly detection

Functions:
    - KnowledgeDiscovery: Main discovery engine
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import json


class KnowledgeDiscovery:
    """
    Extract meaningful insights from unsupervised learning results.
    
    Combines clustering and topic modeling to identify:
    - Semantic relationships between clusters
    - Topic dominance patterns
    - Anomalous documents
    - Legal domain patterns
    """
    
    def __init__(self, texts: List[str], 
                 cluster_labels: np.ndarray,
                 topic_labels: np.ndarray,
                 embeddings: np.ndarray):
        """Initialize knowledge discovery with strict alignment checks."""
        self.texts = texts
        self.cluster_labels = cluster_labels
        self.topic_labels = np.asarray(topic_labels)
        self.embeddings = embeddings
        self.n_docs = len(texts)

        # DEBUG / sanity checks (helps catch the exact failure you saw)
        print("DEBUG: KnowledgeDiscovery init")
        print("DEBUG - Texts:", len(self.texts))
        print("DEBUG - Clusters:", len(self.cluster_labels))
        print("DEBUG - Topics:", len(self.topic_labels))
        
        if len(self.cluster_labels) != len(self.topic_labels):
            raise ValueError(
                f"Topic labels ({len(self.topic_labels)}) and cluster labels ({len(self.cluster_labels)}) "
                f"must match. Texts={len(self.texts)}"
            )
        if len(self.cluster_labels) != len(self.texts):
            raise ValueError(
                f"Cluster labels ({len(self.cluster_labels)}) and texts ({len(self.texts)}) must match."
            )

        print("📖 Initialized knowledge discovery")
        print(f"   Documents: {self.n_docs}")

    
    def get_cluster_summary(self, cluster_id: int, 
                           n_keywords: int = 10) -> Dict:
        """
        Generate summary for a specific cluster.
        
        Args:
            cluster_id (int): Cluster ID
            n_keywords (int): Number of top keywords
            
        Returns:
            Dict: Cluster summary with topics, size, keywords
        """
        # Get cluster documents
        cluster_mask = self.cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_texts = [self.texts[i] for i in cluster_indices]
        cluster_topics = self.topic_labels[cluster_indices]
        
        # Topic distribution in cluster
        unique_topics, counts = np.unique(cluster_topics, return_counts=True)
        topic_distribution = {
            int(t): {
                "count": int(c),
                "percentage": float(c / len(cluster_topics) * 100)
            }
            for t, c in zip(unique_topics, counts)
        }
        
        # Extract keywords from cluster texts
        from collections import Counter
        import re
        
        # Simple keyword extraction: common words
        all_words = []
        for text in cluster_texts:
            words = re.findall(r'\b[a-z]+\b', text.lower())
            all_words.extend([w for w in words if len(w) > 4])  # Words > 4 chars
        
        word_freq = Counter(all_words)
        top_keywords = [w for w, _ in word_freq.most_common(n_keywords)]
        
        summary = {
            "cluster_id": int(cluster_id),
            "size": int(len(cluster_indices)),
            "percentage": float(len(cluster_indices) / self.n_docs * 100),
            "topic_distribution": topic_distribution,
            "dominant_topic": int(unique_topics[np.argmax(counts)]),
            "keywords": top_keywords,
            "avg_text_length": float(np.mean([len(t) for t in cluster_texts])),
            "sample_doc_idx": int(cluster_indices[0]) if len(cluster_indices) > 0 else -1
        }
        
        return summary
    
    def get_all_cluster_summaries(self, n_keywords: int = 10) -> Dict:
        """
        Generate summaries for all clusters.
        
        Args:
            n_keywords (int): Keywords per cluster
            
        Returns:
            Dict: Summaries for all clusters
        """
        unique_clusters = np.unique(self.cluster_labels)
        summaries = {}
        
        for cluster_id in unique_clusters:
            summaries[int(cluster_id)] = self.get_cluster_summary(cluster_id, n_keywords)
        
        return summaries
    
    def get_cluster_topic_relationship(self) -> Dict:
        """
        Analyze relationships between clusters and topics.
        
        Returns:
            Dict: Cross-tabulation of clusters × topics
        """
        n_clusters = len(np.unique(self.cluster_labels))
        n_topics = len(np.unique(self.topic_labels))
        
        # Create cross-tabulation
        relationship = {}
        
        for cluster_id in range(n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_topics = self.topic_labels[cluster_mask]
            
            topic_counts = {}
            for topic_id in np.unique(cluster_topics):
                count = np.sum(cluster_topics == topic_id)
                topic_counts[int(topic_id)] = int(count)
            
            relationship[int(cluster_id)] = topic_counts
        
        return relationship
    
    def find_pure_clusters(self, purity_threshold: float = 0.8) -> List[int]:
        """
        Find clusters with high topic purity (dominated by single topic).
        
        Args:
            purity_threshold (float): Minimum purity (0-1)
            
        Returns:
            List[int]: IDs of pure clusters
        """
        pure_clusters = []
        
        unique_clusters = np.unique(self.cluster_labels)
        
        for cluster_id in unique_clusters:
            cluster_mask = self.cluster_labels == cluster_id
            cluster_topics = self.topic_labels[cluster_mask]
            
            # Calculate purity
            unique_topics, counts = np.unique(cluster_topics, return_counts=True)
            max_topic_count = np.max(counts)
            purity = max_topic_count / len(cluster_topics)
            
            if purity >= purity_threshold:
                pure_clusters.append(int(cluster_id))
        
        return pure_clusters
    
    def find_anomalies(self, method: str = "distance", 
                      threshold: float = 2.0) -> Dict:
        """
        Identify anomalous documents in clusters.
        
        Args:
            method (str): "distance" or "topic_mismatch"
            threshold (float): Anomaly threshold
            
        Returns:
            Dict: Anomalous documents with reasons
        """
        anomalies = {}
        
        if method == "distance":
            # Find documents far from cluster centroid
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=len(np.unique(self.cluster_labels)))
            kmeans.fit(self.embeddings)
            
            # Calculate distances
            distances = np.min(kmeans.transform(self.embeddings), axis=1)
            anomaly_mask = distances > threshold
            
            anomaly_indices = np.where(anomaly_mask)[0]
            
            for idx in anomaly_indices:
                anomalies[int(idx)] = {
                    "reason": "far_from_centroid",
                    "distance": float(distances[idx]),
                    "cluster": int(self.cluster_labels[idx])
                }
        
        elif method == "topic_mismatch":
            # Find documents with unusual topic-cluster combinations
            cluster_topic_rel = self.get_cluster_topic_relationship()
            
            for doc_idx in range(self.n_docs):
                cluster_id = self.cluster_labels[doc_idx]
                topic_id = self.topic_labels[doc_idx]
                
                # Check if topic is rare in this cluster
                if int(topic_id) not in cluster_topic_rel[int(cluster_id)]:
                    anomalies[doc_idx] = {
                        "reason": "unusual_topic",
                        "cluster": int(cluster_id),
                        "topic": int(topic_id)
                    }
        
        return anomalies
    
    def get_legal_patterns(self) -> Dict:
        """
        Extract legal domain patterns from results.
        
        Returns:
            Dict: Discovered patterns and insights
        """
        patterns = {
            "high_purity_clusters": self.find_pure_clusters(),
            "anomalies": self.find_anomalies(),
            "cluster_topic_distribution": self.get_cluster_topic_relationship(),
            "cluster_summaries": self.get_all_cluster_summaries()
        }
        
        return patterns
    
    def get_insights(self) -> List[Dict]:
        """
        Generate human-readable insights from the data.
        
        Returns:
            List[Dict]: List of insights
        """
        insights = []
        
        # Insight 1: Dominant clusters
        cluster_sizes = {}
        for cluster_id in np.unique(self.cluster_labels):
            size = np.sum(self.cluster_labels == cluster_id)
            cluster_sizes[int(cluster_id)] = int(size)
        
        largest_cluster = max(cluster_sizes, key=cluster_sizes.get)
        largest_size = cluster_sizes[largest_cluster]
        
        insights.append({
            "type": "dominant_cluster",
            "cluster_id": largest_cluster,
            "size": largest_size,
            "percentage": float(largest_size / self.n_docs * 100),
            "description": f"Cluster {largest_cluster} contains {largest_size} documents ({largest_size/self.n_docs*100:.1f}%)"
        })
        
        # Insight 2: Topic concentration
        topic_counts = {}
        for topic_id in np.unique(self.topic_labels):
            if topic_id != -1:
                count = np.sum(self.topic_labels == topic_id)
                topic_counts[int(topic_id)] = int(count)
        
        if topic_counts:
            dominant_topic = max(topic_counts, key=topic_counts.get)
            dominant_count = topic_counts[dominant_topic]
            
            insights.append({
                "type": "dominant_topic",
                "topic_id": dominant_topic,
                "count": dominant_count,
                "percentage": float(dominant_count / self.n_docs * 100),
                "description": f"Topic {dominant_topic} appears in {dominant_count} documents ({dominant_count/self.n_docs*100:.1f}%)"
            })
        
        # Insight 3: Pure clusters
        pure_clusters = self.find_pure_clusters()
        insights.append({
            "type": "pure_clusters",
            "count": len(pure_clusters),
            "cluster_ids": pure_clusters,
            "description": f"{len(pure_clusters)} clusters show high topic purity"
        })
        
        # Insight 4: Anomalies
        anomalies = self.find_anomalies()
        insights.append({
            "type": "anomalies",
            "count": len(anomalies),
            "percentage": float(len(anomalies) / self.n_docs * 100),
            "description": f"{len(anomalies)} anomalous documents detected ({len(anomalies)/self.n_docs*100:.1f}%)"
        })
        
        return insights
    
    def to_dict(self) -> Dict:
        """
        Convert all discoveries to dictionary format.
        
        Returns:
            Dict: Complete knowledge discovery results
        """
        return {
            "cluster_summaries": self.get_all_cluster_summaries(),
            "cluster_topic_relationship": self.get_cluster_topic_relationship(),
            "legal_patterns": self.get_legal_patterns(),
            "insights": self.get_insights()
        }
