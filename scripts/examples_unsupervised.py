"""
Unsupervised Learning Examples
================================
Practical examples demonstrating the complete pipeline for legal document analysis.

Each example shows how to use individual modules and the complete orchestrator.

Examples:
1. Basic embeddings generation
2. Clustering and optimal K-finding
3. Topic discovery
4. Knowledge extraction
5. Complete end-to-end pipeline
6. Custom visualization
7. Semantic search across documents
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pdf_extractor import PDFExtractor
from preprocessing import TextPreprocessor
from chunking import DocumentChunker
from feature_extraction import EmbeddingGenerator
from clustering import ClusterAnalyzer
from topic_modeling import TopicModeler
from knowledge_discovery import KnowledgeDiscovery
from storage import StorageManager
from visualization import VisualizationManager
from unsupervised_app import UnsupervisedAnalyzer


# ============================================================================
# Example 1: Basic Embeddings Generation
# ============================================================================

def example_1_embeddings():
    """
    Generate semantic embeddings from legal text chunks.
    
    Shows:
    - Loading embedding model
    - Embedding single texts
    - Batch embedding with progress
    - Calculating similarity
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: BASIC EMBEDDINGS GENERATION")
    print("=" * 70)
    
    # Sample legal texts
    texts = [
        "Employment agreement between Company ABC and John Smith effective January 1, 2024.",
        "Service Level Agreement specifying uptime requirements and service credits.",
        "Non-disclosure agreement protecting confidential business information.",
        "License agreement granting usage rights for software products.",
        "Partnership agreement defining roles, responsibilities, and profit sharing.",
    ]
    
    # Initialize generator
    generator = EmbeddingGenerator("all-MiniLM-L6-v2")
    
    # Generate embeddings
    print("\n📊 Generating embeddings...")
    embeddings = generator.embed_batch(texts, batch_size=32, show_progress=True)
    
    # Calculate similarities
    print("\n🔍 Calculating similarities...")
    print(f"\nSimilarity between 'Employment agreement' and 'Partnership agreement':")
    sim = generator.get_similarity(embeddings[0], embeddings[4])
    print(f"   Similarity: {sim:.3f}")
    
    # Semantic search
    print("\n🔎 Semantic search - Find documents similar to 'confidential information':")
    query = "confidential information"
    results = generator.get_top_similar(query, texts, embeddings, top_k=3)
    for i, (text, similarity) in enumerate(results, 1):
        print(f"   {i}. {text[:50]}... (similarity: {similarity:.3f})")
    
    return embeddings, texts


# ============================================================================
# Example 2: Clustering with Optimal K-Finding
# ============================================================================

def example_2_clustering(embeddings, texts):
    """
    Find optimal number of clusters and perform clustering.
    
    Shows:
    - Finding optimal K using Silhouette Score
    - Fitting clusters
    - Getting cluster statistics
    - Visualizing clusters
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: CLUSTERING WITH OPTIMAL K-FINDING")
    print("=" * 70)
    
    # Initialize cluster analyzer
    analyzer = ClusterAnalyzer(embeddings)
    
    # Find optimal K
    print("\n🔍 Finding optimal number of clusters...")
    optimal_k = analyzer.find_optimal_clusters(k_range=(2, 5))
    print(f"   Optimal K: {optimal_k}")
    
    # Fit clusters
    print(f"\n📊 Fitting {optimal_k} clusters...")
    analyzer.fit(optimal_k)
    
    # Get statistics
    stats = analyzer.get_statistics()
    print(f"\n📈 Cluster Statistics:")
    print(f"   Silhouette Score: {stats['silhouette_score']:.3f}")
    print(f"   Inertia: {stats['inertia']:.1f}")
    print(f"   Davies-Bouldin Index: {stats.get('davies_bouldin_index', 'N/A')}")
    
    # Get cluster assignments
    assignments = analyzer.get_cluster_assignments()
    for cluster_id, indices in assignments.items():
        print(f"\n   Cluster {cluster_id}: {len(indices)} documents")
        for idx in indices:
            print(f"      • {texts[idx][:50]}...")
    
    # Get closest documents to centroids
    print(f"\n🎯 Most representative documents per cluster:")
    for cluster_id in range(optimal_k):
        closest = analyzer.get_closest_documents(cluster_id, n=1)
        if closest:
            idx, _dist = closest[0]  # (doc_index, distance)
            print(f"   Cluster {cluster_id}: {texts[idx][:50]}...")

    
    return analyzer


# ============================================================================
# Example 3: Topic Discovery
# ============================================================================

def example_3_topics(embeddings):
    """
    Discover hidden topics in documents.
    
    Shows:
    - Fitting BERTopic model
    - Getting topic information
    - Finding top topics
    - Getting document-topic assignments
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: TOPIC DISCOVERY")
    print("=" * 70)
    
    # Initialize topic modeler
    modeler = TopicModeler(embeddings)
    
    # Fit model
    print("\n🔍 Fitting BERTopic model...")
    modeler.fit(language="english")
    
    # Reduce outliers
    print("\n🔧 Reducing outliers...")
    modeler.reduce_outliers()
    
    # Get statistics
    stats = modeler.get_statistics()
    print(f"\n📊 Topic Statistics:")
    print(f"   Topics: {stats['n_topics']}")
    print(f"   Outliers: {stats['n_outliers']}")
    print(f"   Total documents: {stats['n_documents']}")
    
    # Get topic information
    print(f"\n📚 Top Topics:")
    topic_info = modeler.get_topic_info()
    top_topics = modeler.get_top_topics(n=3)
    
    for topic_id, info in top_topics.items():
        print(f"\n   Topic {topic_id}: {info['label']}")
        print(f"      Documents: {info['document_count']} ({info['document_percentage']:.1f}%)")
        print(f"      Keywords: {', '.join(info['keywords'][:5])}")
    
    return modeler


# ============================================================================
# Example 4: Knowledge Extraction
# ============================================================================

def example_4_knowledge(texts, cluster_analyzer, topic_modeler, embeddings):
    """
    Extract actionable insights from clustering and topics.
    
    Shows:
    - Getting cluster summaries
    - Finding pure clusters
    - Detecting anomalies
    - Extracting business insights
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: KNOWLEDGE EXTRACTION & INSIGHTS")
    print("=" * 70)
    
    # Initialize knowledge discovery
    discoverer = KnowledgeDiscovery(
        texts,
        cluster_analyzer.labels,
        topic_modeler.topics,
        embeddings
    )
    
    # Get cluster summaries
    print("\n📋 Cluster Summaries:")
    summaries = discoverer.get_all_cluster_summaries(n_keywords=5)
    for cluster_id, summary in list(summaries.items())[:2]:
        print(f"\n   Cluster {cluster_id}:")
        print(f"      Size: {summary['size']} documents ({summary['percentage']:.1f}%)")
        print(f"      Dominant topic: {summary['dominant_topic']}")
        print(f"      Keywords: {', '.join(summary['keywords'][:3])}")
    
    # Find pure clusters
    print("\n\n🎯 Pure Clusters (high topic coherence):")
    pure_clusters = discoverer.find_pure_clusters(purity_threshold=0.8)
    print(f"   Found {len(pure_clusters)} pure clusters: {pure_clusters}")
    
    # Detect anomalies
    print("\n\n⚠️  Anomalous Documents:")
    anomalies = discoverer.find_anomalies(method="topic_mismatch")
    if anomalies:
        for doc_idx, info in list(anomalies.items())[:3]:
            print(f"   Doc {doc_idx}: {info['reason']} (Cluster {info['cluster']}, Topic {info['topic']})")
    else:
        print("   No anomalies detected!")
    
    # Get insights
    print("\n\n💡 Key Insights:")
    insights = discoverer.get_insights()
    for insight in insights:
        print(f"\n   {insight['type'].upper()}:")
        print(f"      {insight['description']}")
    
    return discoverer


# ============================================================================
# Example 5: Complete End-to-End Pipeline
# ============================================================================

def example_5_complete_pipeline():
    """
    Run the complete unsupervised learning pipeline on PDFs.
    
    Shows:
    - Loading PDFs
    - Preprocessing
    - Chunking
    - Embeddings
    - Clustering
    - Topics
    - Knowledge discovery
    - Saving results
    - Generating visualizations
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: COMPLETE END-TO-END PIPELINE")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = UnsupervisedAnalyzer(
        output_folder="processed_data",
        pdf_folder="processed_data",
        chunk_size=512,
        n_clusters=None  # Auto-detect
    )
    
    # Run analysis
    analyzer.run_analysis()
    
    print("\n✓ Pipeline complete! Check 'processed_data' folder for results.")


# ============================================================================
# Example 6: Custom Visualization
# ============================================================================

def example_6_visualization(embeddings, cluster_labels, modeler, discoverer):
    """
    Create custom visualizations from analysis results.
    
    Shows:
    - 2D cluster visualization
    - Topic distribution charts
    - Cluster-topic heatmap
    - Summary dashboard
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: CUSTOM VISUALIZATIONS")
    print("=" * 70)
    
    visualizer = VisualizationManager("processed_data")
    
    # Get topic information
    topic_info = modeler.get_topic_info()
    discoverer_obj = discoverer
    cluster_topic_rel = discoverer_obj.get_cluster_topic_relationship()
    
    print("\n📊 Creating visualizations...")
    
    # 2D cluster plot
    print("   • Creating 2D cluster visualization...")
    visualizer.plot_cluster_2d(embeddings, cluster_labels)
    
    # Topic distribution
    print("   • Creating topic distribution chart...")
    visualizer.plot_topic_distribution(topic_info)
    
    # Cluster-topic heatmap
    print("   • Creating cluster-topic heatmap...")
    visualizer.plot_cluster_topic_heatmap(cluster_topic_rel)
    
    # Cluster sizes
    print("   • Creating cluster sizes chart...")
    visualizer.plot_cluster_sizes(cluster_labels)
    
    print("\n✓ Visualizations saved to processed_data/")


# ============================================================================
# Example 7: Semantic Search
# ============================================================================

def example_7_semantic_search(texts, embeddings):
    """
    Perform semantic search across document collection.
    
    Shows:
    - Searching for similar documents
    - Finding documents by semantic meaning
    - Building search applications
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: SEMANTIC SEARCH")
    print("=" * 70)
    
    generator = EmbeddingGenerator()
    
    # Search queries
    queries = [
        "confidentiality and non-disclosure",
        "payment terms and conditions",
        "liability and indemnification"
    ]
    
    print("\n🔎 Semantic search results:\n")
    for query in queries:
        print(f"Query: '{query}'")
        results = generator.get_top_similar(query, texts, embeddings, top_k=2)
        for i, (text, sim) in enumerate(results, 1):
            print(f"   {i}. {text[:60]}... (similarity: {sim:.3f})")
        print()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run selected examples."""
    print("\n" + "=" * 70)
    print("UNSUPERVISED LEARNING EXAMPLES FOR SMART LEGAL ASSISTANCE")
    print("=" * 70)
    
    print("\n📖 Available Examples:")
    print("   1. Basic Embeddings Generation")
    print("   2. Clustering with Optimal K-Finding")
    print("   3. Topic Discovery")
    print("   4. Knowledge Extraction & Insights")
    print("   5. Complete End-to-End Pipeline")
    print("   6. Custom Visualizations (requires 1-4)")
    print("   7. Semantic Search")
    print("   8. Run All Examples")
    
    choice = input("\n👉 Select example (1-8) or press Enter for all: ").strip() or "8"
    
    # Run examples
    if choice in ["1", "8"]:
        embeddings, texts = example_1_embeddings()
    
    if choice in ["2", "8"]:
        embeddings, texts = example_1_embeddings()
        cluster_analyzer = example_2_clustering(embeddings, texts)
    
    if choice in ["3", "8"]:
        embeddings, texts = example_1_embeddings()
        topic_modeler = example_3_topics(embeddings)
    
    if choice in ["4", "8"]:
        embeddings, texts = example_1_embeddings()
        cluster_analyzer = example_2_clustering(embeddings, texts)
        topic_modeler = example_3_topics(embeddings)
        discoverer = example_4_knowledge(texts, cluster_analyzer, topic_modeler, embeddings)
    
    if choice == "5":
        example_5_complete_pipeline()
    
    if choice in ["6", "8"]:
        embeddings, texts = example_1_embeddings()
        cluster_analyzer = example_2_clustering(embeddings, texts)
        topic_modeler = example_3_topics(embeddings)
        discoverer = example_4_knowledge(texts, cluster_analyzer, topic_modeler, embeddings)
        example_6_visualization(embeddings, cluster_analyzer.labels, topic_modeler, discoverer)
    
    if choice in ["7", "8"]:
        embeddings, texts = example_1_embeddings()
        example_7_semantic_search(texts, embeddings)
    
    print("\n" + "=" * 70)
    print("✓ Examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
