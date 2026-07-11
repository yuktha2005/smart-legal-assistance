# Unsupervised Learning Pipeline - Smart Legal Assistance

## Overview

The Unsupervised Learning pipeline automatically discovers patterns, topics, and document groupings in legal documents **without manual labeling or annotation**.

### What It Does

```
Legal PDFs
    ↓
Text Extraction (PyMuPDF + OCR fallback)
    ↓
Preprocessing (Spacy lemmatization, citation removal)
    ↓
Chunking (512-char sliding windows)
    ↓
Semantic Embeddings (384-dimensional vectors)
    ↓
Clustering (K-Means with auto K-detection)
    ↓
Topic Modeling (BERTopic with auto labels)
    ↓
Knowledge Discovery (Insights, patterns, anomalies)
    ↓
Results (JSON, visualizations, embeddings)
```

---

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download spacy model for lemmatization
python -m spacy download en_core_web_sm

# Install Tesseract for scanned PDFs (Windows)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. Run the Pipeline

```bash
# Quick run (uses sample data)
python unsupervised_app.py

# Custom configuration
from unsupervised_app import UnsupervisedAnalyzer

analyzer = UnsupervisedAnalyzer(
    output_folder="my_results",
    pdf_folder="my_pdfs",
    chunk_size=512,
    n_clusters=5  # or None for auto-detect
)
analyzer.run_analysis()
```

### 3. View Results

Results are saved in `processed_data/`:

```
processed_data/
├── embeddings.npy                    # 384-dim vectors (binary)
├── cluster_results.json              # Clustering assignments + stats
├── topic_results.json                # Topic keywords + labels
├── processed_chunks.json             # Text chunks + metadata
├── knowledge_discovery.json          # Insights + patterns
├── cluster_visualization.png         # 2D PCA scatter plot
├── cluster_sizes.png                 # Distribution chart
├── topic_distribution.png            # Topic breakdown
├── cluster_topic_heatmap.png         # Relationship matrix
└── summary_dashboard.png             # 4-panel overview
```

---

## Architecture

### 10-Phase Pipeline

#### Phase 1: Document Extraction

- Extracts text from PDF files
- Falls back to OCR (Tesseract) for scanned PDFs
- Handles both text-based and image-based documents

#### Phase 2: Text Preprocessing

- Converts to lowercase
- Removes legal citations (e.g., "Smith v. Jones")
- Removes footnotes and reference markers
- Normalizes whitespace

#### Phase 3: Document Chunking

- Splits documents into 512-character chunks
- Maintains 64-character overlap for context preservation
- Preserves metadata (source document, position)
- Implements intelligent sentence-boundary breaking

#### Phase 4: Semantic Embeddings

- Generates 384-dimensional vectors per chunk
- Uses `all-MiniLM-L6-v2` model (fast, high quality)
- Captures semantic meaning for clustering
- Memory efficient (32 batch processing)

#### Phase 5: Clustering

- K-Means clustering with auto K-determination
- Tests K values from 2 to 10
- Uses Silhouette Score for optimal K selection
- Alternative: Elbow Method and Davies-Bouldin Index

#### Phase 6: Topic Modeling

- BERTopic for unsupervised topic discovery
- Auto-generates interpretable topic labels
- Reduces outlier documents
- Computes topic probabilities per document

#### Phase 7: Knowledge Extraction

- Identifies cluster summaries with keywords
- Finds pure clusters (high topic coherence)
- Detects anomalous documents
- Generates business insights

#### Phase 8: Results Persistence

- Saves embeddings (.npy binary format)
- Saves all results as JSON (human-readable)
- Includes metadata and version info
- Enables reproducible analysis

#### Phase 9: Visualization Generation

- 2D PCA scatter plot of clusters
- Topic distribution bar/pie charts
- Cluster size distribution
- Cluster-topic relationship heatmap
- 4-panel summary dashboard

#### Phase 10: Report Generation

- Comprehensive analysis summary
- Key statistics and metrics
- Top insights from analysis
- Metadata for tracking

---

## Module Reference

### 1. `feature_extraction.py`

**Purpose**: Generate semantic embeddings from text

```python
from src.feature_extraction import EmbeddingGenerator

# Initialize
generator = EmbeddingGenerator("all-MiniLM-L6-v2")

# Single text
embedding = generator.embed_single("legal text here")  # (384,)

# Batch processing
embeddings = generator.embed_batch(texts, batch_size=32)  # (n_texts, 384)

# Similarity
sim = generator.get_similarity(emb1, emb2)  # 0.0 to 1.0

# Semantic search
results = generator.get_top_similar("query", chunks, embeddings, top_k=5)
# Returns: [(text, similarity), ...]

# Persistence
generator.save_embeddings(embeddings, "embeddings.npy")
loaded = generator.load_embeddings("embeddings.npy")
```

**Key Methods**:

- `embed_single(text)` - Single text embedding
- `embed_batch(texts, batch_size, show_progress)` - Batch processing
- `get_similarity(emb1, emb2)` - Cosine similarity
- `get_top_similar(query, chunks, embeddings, top_k)` - Search
- `save_embeddings()`, `load_embeddings()` - Persistence

**Model Choice**:

- `all-MiniLM-L6-v2` (default): Fast, balanced (384-dim, 33MB)
- `all-mpnet-base-v2`: Higher quality, slower (768-dim, 420MB)

---

### 2. `clustering.py`

**Purpose**: Unsupervised clustering with optimal K detection

```python
from src.clustering import ClusterAnalyzer

# Initialize with embeddings
analyzer = ClusterAnalyzer(embeddings)  # shape: (n_docs, 384)

# Find optimal K (uses Silhouette Score)
optimal_k = analyzer.find_optimal_clusters(k_range=(2, 10))  # Default

# Fit clusters
analyzer.fit(optimal_k)

# Get statistics
stats = analyzer.get_statistics()
print(f"Silhouette Score: {stats['silhouette_score']:.3f}")
print(f"Inertia: {stats['inertia']:.0f}")

# Get assignments
assignments = analyzer.get_cluster_assignments()
# Returns: {cluster_id: [doc_indices]}

# Representative documents
closest = analyzer.get_closest_documents(cluster_id=0, n=5)
# Returns: [doc_index, ...]

# Visualizations
analyzer.visualize_clusters(texts, "clusters.png")
analyzer.visualize_elbow("elbow.png")

# Export
results = analyzer.to_dict()
```

**Key Methods**:

- `find_optimal_clusters(k_range)` - Auto K using Silhouette
- `fit(n_clusters)` - Perform clustering
- `get_statistics()` - Metrics
- `get_cluster_assignments()` - Document assignments
- `get_closest_documents()` - Representative docs
- `visualize_clusters()`, `visualize_elbow()` - Plots
- `to_dict()` - Serialization

**Statistics**:

- Silhouette Score (-1 to 1, higher is better)
- Inertia (sum of squared distances)
- Davies-Bouldin Index (lower is better)
- Per-cluster percentages

---

### 3. `topic_modeling.py`

**Purpose**: Discover hidden topics with auto-labeling

```python
from src.topic_modeling import TopicModeler

# Initialize with embeddings
modeler = TopicModeler(embeddings)

# Fit BERTopic
modeler.fit(language="english")

# Reduce outliers
modeler.reduce_outliers()

# Get topic info
topic_info = modeler.get_topic_info()
# Returns: {topic_id: {label, keywords, weights, count, percentage}}

# Top topics
top_topics = modeler.get_top_topics(n=5)

# Document-topic assignment
doc_topics = modeler.get_document_topics(doc_index=0)
# Returns: {primary_topic, top_topics: [{topic_id, probability}, ...]}

# Topic distribution
distribution = modeler.get_topic_distribution()
# Returns: {topic_id: {count, percentage}}

# Statistics
stats = modeler.get_statistics()

# Persistence
modeler.save_model("model_path")
modeler.load_model("model_path")

# Export
results = modeler.to_dict()
```

**Key Methods**:

- `fit(language)` - Discover topics
- `reduce_outliers()` - Improve coherence
- `get_topic_info()` - Topic details
- `get_top_topics()` - Top N topics
- `get_document_topics()` - Doc assignment
- `get_topic_distribution()` - Distribution
- `save_model()`, `load_model()` - Persistence
- `to_dict()` - Serialization

---

### 4. `clustering.py` + `topic_modeling.py` → `knowledge_discovery.py`

**Purpose**: Extract insights from clustering & topics

```python
from src.knowledge_discovery import KnowledgeDiscovery

discoverer = KnowledgeDiscovery(
    texts=chunk_texts,
    cluster_labels=cluster_labels,
    topic_labels=topic_labels,
    embeddings=embeddings
)

# Cluster summaries
summaries = discoverer.get_all_cluster_summaries(n_keywords=10)
# Returns: {cluster_id: {size, keywords, topics, sample_doc}}

# Cluster-topic relationships
relationships = discoverer.get_cluster_topic_relationship()
# Returns: {cluster_id: {topic_id: count}}

# Pure clusters (high coherence)
pure_clusters = discoverer.find_pure_clusters(purity_threshold=0.8)
# Returns: [cluster_ids]

# Anomalous documents
anomalies = discoverer.find_anomalies(method="topic_mismatch")
# Returns: {doc_id: {reason, cluster, topic}}

# Insights
insights = discoverer.get_insights()
# Returns: [{type, description, value}, ...]

# Export
results = discoverer.to_dict()
```

**Key Methods**:

- `get_cluster_summary()` - Single cluster analysis
- `get_all_cluster_summaries()` - All clusters
- `get_cluster_topic_relationship()` - Cross-tabulation
- `find_pure_clusters()` - High coherence clusters
- `find_anomalies()` - Outlier detection
- `get_legal_patterns()` - Domain patterns
- `get_insights()` - Business insights
- `to_dict()` - Serialization

**Insight Types**:

- `dominant_cluster` - Largest cluster
- `dominant_topic` - Most common topic
- `pure_clusters` - Coherent groups
- `anomalies` - Unusual documents

---

### 5. `storage.py`

**Purpose**: Persistent storage for all results

```python
from src.storage import StorageManager

storage = StorageManager("output_folder")

# Save embeddings (binary)
storage.save_embeddings(embeddings, "embeddings.npy")
loaded_emb = storage.load_embeddings("embeddings.npy")

# Save chunks
storage.save_chunks(chunk_texts, metadata_list)

# Save results
storage.save_clustering_results(cluster_results)
storage.save_topic_results(topic_results)
storage.save_knowledge_discovery(discovery_results)

# Save summary
storage.save_summary(summary_dict)

# Load JSON
data = storage.load_json("cluster_results.json")

# List files
files = storage.list_files()

# Get path
filepath = storage.get_file_path("embeddings.npy")
```

---

### 6. `visualization.py`

**Purpose**: Publication-quality visualizations

```python
from src.visualization import VisualizationManager

viz = VisualizationManager("output_folder")

# 2D cluster scatter (PCA)
viz.plot_cluster_2d(embeddings, cluster_labels)

# Topic distribution
viz.plot_topic_distribution(topic_info)

# Elbow curve
viz.plot_elbow_curve(inertias, silhouette_scores, optimal_k)

# Cluster-topic heatmap
viz.plot_cluster_topic_heatmap(cluster_topic_relationships)

# Cluster sizes
viz.plot_cluster_sizes(cluster_labels)

# Summary dashboard (4-panel)
viz.create_summary_dashboard(
    embeddings, cluster_labels,
    topic_info, cluster_topic_rel
)
```

---

### 7. `preprocessing.py`

**Purpose**: Text cleaning and lemmatization

```python
from src.preprocessing import TextPreprocessor

# Initialize (auto-downloads spacy model)
preprocessor = TextPreprocessor("en_core_web_sm")

# Clean text (lemmatization, citation removal)
cleaned = preprocessor.clean_text(raw_text)

# Get sentences
sentences = preprocessor.get_sentences(text)

# Get lemmatized tokens
tokens = preprocessor.get_tokens(text)

# Extract named entities
entities = preprocessor.get_entities(text)
# Returns: {PERSON: [...], ORG: [...], GPE: [...]}

# Statistics
stats = preprocessor.get_statistics(text)
# Returns: {char_count, sentence_count, token_count, ...}
```

**Features**:

- Lemmatization (word form reduction)
- Legal citation removal (case names, citations)
- Stopword removal (customizable)
- Legal keyword preservation
- Named entity extraction
- Sentence tokenization

---

### 8. `chunking.py`

**Purpose**: Break documents into overlapping chunks

```python
from src.chunking import DocumentChunker

chunker = DocumentChunker(
    chunk_size=512,
    overlap_size=64,
    separator="\n"
)

# Chunk single document
chunks = chunker.chunk_document(
    text="long legal document...",
    doc_id="doc_001",
    doc_source="contract.pdf"
)

# Chunk multiple documents
all_chunks = chunker.chunk_documents([
    (text1, "doc_001", "source1"),
    (text2, "doc_002", "source2"),
])

# Statistics
stats = chunker.get_chunk_statistics(chunks)
# Returns: {total_chunks, avg_length, min_length, max_length, ...}

# Get context around chunk
context = chunker.extract_chunk_context(chunks, chunk_idx=5, context_size=2)
# Returns: {main_chunk, context_chunks, combined_text}

# Validate chunks
validation = chunker.validate_chunks(chunks)
# Returns: {valid, issues, problematic_chunks}
```

---

### 9. `ocr_handler.py`

**Purpose**: OCR for scanned PDFs

```python
from src.ocr_handler import OCRHandler

ocr = OCRHandler(language="eng", tesseract_path=None)

# Detect if PDF is scanned
is_scanned, ratio = ocr.is_scanned_pdf("document.pdf", sample_pages=3)

# Extract with OCR
texts = ocr.extract_text_with_ocr("document.pdf")

# Hybrid extraction (PyMuPDF + OCR fallback)
result = ocr.hybrid_extraction(
    "document.pdf",
    ocr_threshold=0.1
)
# Returns: {method, texts, text_ratio, fallback_used}

# OCR confidence metrics
confidence = ocr.get_ocr_confidence(image_path="page.png")
# Returns: {avg_confidence, max_confidence, words_detected, ...}

# Preprocess image for better OCR
from PIL import Image
image = Image.open("page.png")
processed = ocr.preprocess_image_for_ocr(image)
```

---

## Configuration

### Model Selection

| Aspect     | Setting                                                       |
| ---------- | ------------------------------------------------------------- |
| Embeddings | `all-MiniLM-L6-v2` (384-dim) or `all-mpnet-base-v2` (768-dim) |
| Spacy      | `en_core_web_sm` (fast) or `en_core_web_lg` (accurate)        |
| Clustering | K-Means, auto K from 2-10                                     |
| Topics     | BERTopic with HDBSCAN                                         |
| Chunking   | 512 chars with 64-char overlap                                |
| OCR        | Tesseract (optional, auto-detected)                           |

### Hyperparameters

```python
# Embedding batch size
batch_size = 32  # Adjust for memory constraints

# Clustering
k_range = (2, 10)  # Test range for optimal K
min_samples = 5    # HDBSCAN parameter

# Topics
language = "english"
nr_topics = "auto"  # Let BERTopic decide

# Chunking
chunk_size = 512
overlap_size = 64

# Visualizations
dpi = 300           # Resolution
figsize = (14, 8)   # Figure size
colormap = "Set3"   # Color scheme
```

---

## Common Tasks

### 1. Analyze Custom PDFs

```python
from unsupervised_app import UnsupervisedAnalyzer

analyzer = UnsupervisedAnalyzer(
    output_folder="my_results",
    pdf_folder="/path/to/pdfs"
)
analyzer.run_analysis()
```

### 2. Semantic Search

```python
from src.feature_extraction import EmbeddingGenerator

generator = EmbeddingGenerator()
embeddings = generator.embed_batch(documents)

query_results = generator.get_top_similar(
    "search query",
    documents,
    embeddings,
    top_k=10
)
```

### 3. Re-analyze with Different K

```python
# Load existing embeddings
embeddings = np.load("processed_data/embeddings.npy")

# Try different K values
for k in [3, 4, 5, 6]:
    analyzer = ClusterAnalyzer(embeddings)
    analyzer.fit(k)
    stats = analyzer.get_statistics()
    print(f"K={k}: Silhouette={stats['silhouette_score']:.3f}")
```

### 4. Extract Specific Insights

```python
discoverer = KnowledgeDiscovery(texts, cluster_labels, topics, embeddings)

# Pure clusters only
pure = discoverer.find_pure_clusters(purity_threshold=0.9)

# Anomalies
anomalies = discoverer.find_anomalies()

# Top cluster summaries
summaries = discoverer.get_all_cluster_summaries()
sorted_summaries = sorted(
    summaries.items(),
    key=lambda x: x[1]['size'],
    reverse=True
)
```

---

## Performance Tips

1. **Batch Processing**: Use `batch_size=32` for embeddings
2. **Memory**: For 1000+ docs, use GPU or stream processing
3. **Clustering**: Auto K-detection can be slow; cache results
4. **Visualization**: Save to disk instead of displaying
5. **OCR**: Only run on scanned PDFs; detect first

---

## Output Files Explained

| File                        | Format         | Purpose                         |
| --------------------------- | -------------- | ------------------------------- |
| `embeddings.npy`            | Binary (NumPy) | All document embeddings         |
| `cluster_results.json`      | JSON           | Cluster assignments, statistics |
| `topic_results.json`        | JSON           | Topic keywords, probabilities   |
| `processed_chunks.json`     | JSON           | Text chunks with metadata       |
| `knowledge_discovery.json`  | JSON           | Summaries, insights, patterns   |
| `cluster_visualization.png` | PNG            | 2D scatter plot of clusters     |
| `topic_distribution.png`    | PNG            | Bar/pie charts of topics        |
| `cluster_topic_heatmap.png` | PNG            | Relationship matrix             |
| `summary_dashboard.png`     | PNG            | 4-panel overview                |

---

## Troubleshooting

### Issue: "Spacy model not found"

```bash
python -m spacy download en_core_web_sm
```

### Issue: "Tesseract not found"

- Install from: https://github.com/UB-Mannheim/tesseract/wiki
- Set path: `OCRHandler(tesseract_path="C:/Program Files/Tesseract-OCR/tesseract.exe")`

### Issue: Memory error with large embeddings

```python
# Process in batches
batch_size = 16  # Reduce from 32
embeddings = generator.embed_batch(texts, batch_size=batch_size)
```

### Issue: Poor clustering quality

- Check Silhouette Score (should be > 0.3)
- Try different K range: `find_optimal_clusters(k_range=(3, 8))`
- Increase chunk overlap for better context

---

## Next Steps

1. **Tune parameters** for your specific domain
2. **Validate results** by comparing with manual review
3. **Export to BI tools** (Tableau, Power BI) for visualization
4. **Integrate with workflows** (alerting, reporting)
5. **Expand to other domains** (contracts, regulations, etc.)

---

## References

- [SentenceTransformers](https://www.sbert.net/)
- [BERTopic](https://maartengr.github.io/BERTopic/)
- [Spacy](https://spacy.io/)
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)

---

**Version**: 1.0  
**Last Updated**: 2026-06-19  
**Status**: Production Ready
