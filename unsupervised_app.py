"""
Unsupervised Learning Application
==================================
End-to-end pipeline orchestrator for legal document analysis.

Orchestrates:
1. PDF extraction (with OCR fallback)
2. Text preprocessing with domain-aware lemmatization
3. Document chunking (sliding window)
4. Semantic embedding generation
5. Unsupervised clustering (K-Means with auto K)
6. Topic discovery (BERTopic)
7. Knowledge extraction
8. Persistent storage
9. Comprehensive visualization

Main entry point for the Smart Legal Assistance unsupervised pipeline.

Usage:
    app = UnsupervisedAnalyzer("processed_data", "path/to/pdfs")
    app.run_analysis()
    app.generate_report()

Classes:
    - UnsupervisedAnalyzer: Main orchestrator
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime
import json

# Import all modules
from src.pdf_extractor import PDFExtractor
from src.preprocessing import TextPreprocessor
from src.chunking import DocumentChunker
from src.feature_extraction import EmbeddingGenerator
from src.clustering import ClusterAnalyzer
from src.topic_modeling import TopicModeler
from src.knowledge_discovery import KnowledgeDiscovery
from src.storage import StorageManager
from src.visualization import VisualizationManager
from src.ocr_handler import OCRHandler


class UnsupervisedAnalyzer:
    """
    End-to-end unsupervised learning pipeline for legal documents.
    
    Combines clustering, topic modeling, and knowledge discovery
    to automatically extract patterns from legal documents.
    """
    
    def __init__(self, output_folder: str = "processed_data",
                 pdf_folder: Optional[str] = None,
                 chunk_size: int = 512,
                 n_clusters: Optional[int] = None):
        """
        Initialize the analyzer.
        
        Args:
            output_folder (str): Where to save results
            pdf_folder (str, optional): Where PDFs are located
            chunk_size (int): Size of document chunks
            n_clusters (int, optional): Number of clusters (auto-detect if None)
        """
        self.output_folder = output_folder
        self.pdf_folder = pdf_folder or "processed_data"
        self.chunk_size = chunk_size
        self.n_clusters = n_clusters
        
        # Initialize components
        self.storage = StorageManager(output_folder)
        self.visualizer = VisualizationManager(output_folder)
        
        # Results storage
        self.texts = []
        self.chunks = []
        self.embeddings = None
        self.cluster_analyzer = None
        self.topic_modeler = None
        self.knowledge_discovery = None
        
        print("=" * 60)
        print("🚀 SMART LEGAL ASSISTANCE - UNSUPERVISED ANALYZER")
        print("=" * 60)
        print(f"Output folder: {output_folder}")
        print(f"Chunk size: {chunk_size}")
    
    def extract_documents(self) -> None:
        """Extract text from all PDFs in folder."""
        print("\n" + "=" * 60)
        print("PHASE 1: DOCUMENT EXTRACTION")
        print("=" * 60)
        
        # Find all PDFs
        pdf_files = list(Path(self.pdf_folder).glob("*.pdf"))
        
        extractor = None  # will instantiate per-file below
        
        if not pdf_files:
            # If no PDFs are found, do NOT silently fall back to demo data in production.
            # This protects the “real project” from appearing to work while actually analyzing samples.
            raise FileNotFoundError(
                f"No PDF files found in '{self.pdf_folder}'. "
                "Expected real documents in the 'legal_pdfs' folder. "
                "Add PDFs or configure pdf_folder correctly."
            )
        else:
            print(f"📄 Found {len(pdf_files)} PDF files")
            
            for pdf_file in pdf_files[:5]:  # Limit to first 5 for demo
                try:
                    extractor = PDFExtractor(str(pdf_file))
                    text, _num_pages, _char_count = extractor.extract_all_text()
                    if text.strip():
                        self.texts.append(text)
                        print(f"   ✓ {pdf_file.name}: {len(text)} chars")
                except Exception as e:
                    print(f"   ✗ {pdf_file.name}: {e}")
        
        print(f"\n✓ Extracted {len(self.texts)} documents")
    
    def preprocess_documents(self) -> None:
        """Clean and preprocess extracted text."""
        print("\n" + "=" * 60)
        print("PHASE 2: TEXT PREPROCESSING")
        print("=" * 60)
        
        preprocessor = TextPreprocessor()
        
        preprocessed_texts = []
        for i, text in enumerate(self.texts):
            processed = preprocessor.clean_text(text)
            preprocessed_texts.append(processed)
            print(f"   Doc {i+1}: {len(text)} → {len(processed)} chars")
        
        self.texts = preprocessed_texts
        print(f"\n✓ Preprocessed {len(self.texts)} documents")
    
    def chunk_documents(self) -> None:
        """Break documents into chunks."""
        print("\n" + "=" * 60)
        print("PHASE 3: DOCUMENT CHUNKING")
        print("=" * 60)
        
        chunker = DocumentChunker(chunk_size=self.chunk_size)
        
        all_chunks = []
        for i, text in enumerate(self.texts):
            doc_chunks = chunker.chunk_document(
                text,
                doc_id=f"doc_{i}",
                doc_source=f"document_{i}"
            )
            all_chunks.extend(doc_chunks)
            print(f"   Doc {i+1}: {len(doc_chunks)} chunks")
        
        self.chunks = all_chunks
        
        stats = chunker.get_chunk_statistics(all_chunks)
        print(f"\n✓ Created {stats['total_chunks']} chunks")
        print(f"   Avg length: {stats['avg_length']:.0f} chars")
    
    def generate_embeddings(self) -> None:
        """Generate semantic embeddings for chunks."""
        print("\n" + "=" * 60)
        print("PHASE 4: SEMANTIC EMBEDDING GENERATION")
        print("=" * 60)
        
        chunk_texts = [c['text'] for c in self.chunks]
        
        generator = EmbeddingGenerator()
        embeddings = generator.embed_batch(chunk_texts, batch_size=32, show_progress=True)
        
        self.embeddings = embeddings
        
        print(f"\n✓ Generated embeddings")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Dimension: {embeddings.shape[1]}")
    
    def cluster_documents(self) -> None:
        """Perform unsupervised clustering."""
        print("\n" + "=" * 60)
        print("PHASE 5: UNSUPERVISED CLUSTERING")
        print("=" * 60)
        
        analyzer = ClusterAnalyzer(self.embeddings)
        
        # Find optimal K
        if self.n_clusters is None:
            print("🔍 Finding optimal number of clusters...")
            optimal_k = analyzer.find_optimal_clusters(k_range=(2, 10))
            print(f"   Optimal K: {optimal_k}")
        else:
            optimal_k = self.n_clusters
            print(f"   Using specified K: {optimal_k}")
        
        # Fit clusters
        analyzer.fit(optimal_k)
        
        stats = analyzer.get_statistics()
        print(f"\n✓ Clustering complete")
        print(f"   Silhouette Score: {stats['silhouette_score']:.3f}")
        print(f"   Inertia: {stats['inertia']:.0f}")
        
        self.cluster_analyzer = analyzer
    
    def discover_topics(self) -> None:
        """Discover hidden topics."""
        print("\n" + "=" * 60)
        print("PHASE 6: TOPIC DISCOVERY")
        print("=" * 60)
        
        # TopicModeler expects raw documents when BERTopic fails and it falls back to NMF.
        # Topic modeling must align to chunk-level documents (len(self.chunks))
        chunk_texts = [c["text"] for c in self.chunks]
        modeler = TopicModeler(self.embeddings, documents=chunk_texts)
        modeler.fit(language="english")
        # reduce_outliers exists only for the BERTopic path
        if hasattr(modeler, "reduce_outliers"):
            modeler.reduce_outliers()

        
        stats = modeler.get_statistics()
        print(f"\n✓ Topic modeling complete")
        print(f"   Topics: {stats['n_topics']}")
        print(f"   Outliers: {stats['n_outliers']}")
        
        self.topic_modeler = modeler
    
    def extract_knowledge(self) -> None:
        """Extract actionable knowledge."""
        print("\n" + "=" * 60)
        print("PHASE 7: KNOWLEDGE EXTRACTION")
        print("=" * 60)
        
        chunk_texts = [c['text'] for c in self.chunks]
        cluster_labels = self.cluster_analyzer.labels
        topic_labels = self.topic_modeler.topics
        
        discoverer = KnowledgeDiscovery(
            chunk_texts,
            cluster_labels,
            topic_labels,
            self.embeddings
        )
        
        insights = discoverer.get_insights()
        print(f"\n✓ Knowledge extraction complete")
        print(f"   Insights: {len(insights)}")
        
        for insight in insights:
            print(f"   • {insight['description']}")
        
        self.knowledge_discovery = discoverer
    
    def save_results(self) -> None:
        """Save all results to persistent storage."""
        print("\n" + "=" * 60)
        print("PHASE 8: SAVING RESULTS")
        print("=" * 60)
        
        # Save embeddings
        self.storage.save_embeddings(self.embeddings)
        
        # Save chunks
        self.storage.save_chunks(
            [c['text'] for c in self.chunks],
            [{k: v for k, v in c.items() if k != 'text'} for c in self.chunks]
        )
        
        # Save clustering results
        cluster_results = self.cluster_analyzer.to_dict()
        self.storage.save_clustering_results(cluster_results)
        
        # Save topic results
        topic_results = self.topic_modeler.to_dict()
        self.storage.save_topic_results(topic_results)
        
        # Save knowledge discovery
        knowledge = self.knowledge_discovery.to_dict()
        self.storage.save_knowledge_discovery(knowledge)
        
        print("\n✓ All results saved")
    
    def generate_visualizations(self) -> None:
        """Create comprehensive visualizations."""
        print("\n" + "=" * 60)
        print("PHASE 9: VISUALIZATION GENERATION")
        print("=" * 60)
        
        cluster_labels = self.cluster_analyzer.labels
        topic_info = self.topic_modeler.get_topic_info()
        cluster_topic_rel = self.knowledge_discovery.get_cluster_topic_relationship()
        
        # 2D cluster visualization
        self.visualizer.plot_cluster_2d(
            self.embeddings,
            cluster_labels,
            output_path=os.path.join(self.output_folder, "cluster_visualization.png")
        )
        
        # Topic distribution
        self.visualizer.plot_topic_distribution(topic_info)
        
        # Cluster sizes
        self.visualizer.plot_cluster_sizes(cluster_labels)
        
        # Cluster-topic heatmap
        self.visualizer.plot_cluster_topic_heatmap(cluster_topic_rel)
        
        # Summary dashboard
        self.visualizer.create_summary_dashboard(
            self.embeddings,
            cluster_labels,
            topic_info,
            cluster_topic_rel
        )
        
        print("\n✓ Visualizations complete")
    
    def generate_report(self) -> None:
        """Generate comprehensive analysis report."""
        print("\n" + "=" * 60)
        print("PHASE 10: REPORT GENERATION")
        print("=" * 60)
        
        cluster_stats = self.cluster_analyzer.get_statistics()
        topic_stats = self.topic_modeler.get_statistics()
        insights = self.knowledge_discovery.get_insights()
        
        report = {
            "analysis_date": datetime.now().isoformat(),
            "summary": {
                "total_documents": len(self.texts),
                "total_chunks": len(self.chunks),
                "embedding_dimension": self.embeddings.shape[1]
            },
            "clustering": {
                "n_clusters": cluster_stats.get('n_clusters'),
                "silhouette_score": cluster_stats.get('silhouette_score'),
                "inertia": cluster_stats.get('inertia')
            },
            "topics": {
                "n_topics": topic_stats.get('n_topics'),
                "n_outliers": topic_stats.get('n_outliers')
            },
            "top_insights": [
                {
                    "type": i["type"],
                    "description": i["description"]
                }
                for i in insights[:5]
            ]
        }
        
        self.storage.save_summary(report)
        
        print("\n✓ Report generated and saved")
    
    def run_analysis(self) -> None:
        """Execute the complete analysis pipeline."""
        try:
            self.extract_documents()
            self.preprocess_documents()
            self.chunk_documents()
            self.generate_embeddings()
            self.cluster_documents()
            self.discover_topics()
            self.extract_knowledge()
            self.save_results()
            self.generate_visualizations()
            self.generate_report()
            
            print("\n" + "=" * 60)
            print("✓ ANALYSIS COMPLETE!")
            print("=" * 60)
            self.storage.list_files()
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point."""
    analyzer = UnsupervisedAnalyzer(
        output_folder="processed_data",
        pdf_folder=str(Path(__file__).parent / "legal_pdfs")
    )
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
