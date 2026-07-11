# Smart Legal Assistance - Complete Project Documentation

## 1. Project Overview
**Smart Legal Assistance** is a production-grade, containerized web application designed to bridge the gap between complex legal code and the general public, while also serving as a search and analysis utility for legal professionals (advocates and legal researchers). 

Leveraging Generative Artificial Intelligence (GenAI), Retrieval-Augmented Generation (RAG), and advanced Natural Language Processing (NLP), the system simplifies legal procedures, aids in complaint drafting, parses complex sections, and map criminal acts across statutory shifts. Crucially, the platform includes a dual-flow system:
1. **Interactive RAG Chatbot & Assistant Flow**: An interactive query interface utilizing local LLM execution, dense vector indexing, exact section and crime mapping, heuristic intent classification, cross-encoder reranking, semantic verification (Hallucination Guard), and Text-to-Speech (TTS) synthesis.
2. **Unsupervised Learning & Pattern Discovery Flow**: An offline pipeline that extracts text (via PyMuPDF or Tesseract OCR fallback), preprocesses, chunks, embeds, and executes unsupervised clustering (K-Means with automated Silhouette optimization) and topic modeling (BERTopic) on document corpuses. This enables automated exploration and discovery of hidden themes, business insights, and structural anomalies in unstructured legal texts without manual labels.

> [!IMPORTANT]
> **Disclaimer**: This system is designed and intended strictly for educational, informational, and research purposes. It does not constitute professional legal advice. Users must consult with qualified legal counsel for actual legal proceedings.

---

## 2. Abstract
Modern legal systems are characterized by an overwhelming volume of documents, dense archaic terminology, and frequent statutory evolutions. In India, the transition from the historic Indian Penal Code (IPC) to the modern Bharatiya Nyaya Sanhita (BNS) has further amplified accessibility barriers for common citizens and workload pressures for legal advocates. 

This project implements **Smart Legal Assistance**, an AI-powered assistant built on a modular Python architecture containerized via Docker. The frontend is driven by Streamlit, integrating a local relational database (MySQL managed via SQLAlchemy ORM) for secure user sessions, audit logging, and feedback loops. The core information retrieval is implemented using a hybrid architecture: exact-match section indexes are coupled with dense semantic vector searches using FAISS (`all-MiniLM-L6-v2`) and keyword searches via BM25. User queries are routed through a greedy intent classifier and a cross-encoder reranking layer (`ms-marco-MiniLM-L-6-v2`) to supply relevant context to a localized Large Language Model (Ollama-managed Qwen models) under strict formatting directives. To verify correctness, a post-generation Hallucination Guard calculates keyword overlap between the generated response and the retrieved source material, falling back to a structured warning under low-confidence scenarios. 

Complementing the chatbot is an unsupervised learning pipeline that parses legal corpora page-by-page, dynamically identifies the optimal cluster count $K$, maps out topic keywords using BERTopic, flags anomalous documents, and renders high-resolution visualization dashboards (PCA scatter plots, Heatmaps, and 4-panel dashboards). The addition of cached text-to-speech feedback via Piper TTS ensures accessibility. The resulting application represents a complete, secure, and robust implementation of GenAI and unsupervised learning tailored to the legal domain.

---

## 3. Problem Statement
The accessibility of legal remedies is hindered by three primary challenges:
1. **The Language Barrier of Legal Statutes**: Legal texts are written in highly specialized, formal language (legalese) that is difficult for non-lawyers to interpret. Simple tasks like identifying the punishment for a specific offense or understanding the difference between cognizable and non-cognizable charges require navigating dense volumes of legal code.
2. **Statutory Transition Overhead**: The structural transition from the Indian Penal Code (IPC) to the Bharatiya Nyaya Sanhita (BNS) has created mapping friction. Legal professionals and citizens struggle to map old section numbers (e.g., IPC Section 302 for murder, IPC Section 378 for theft) to their new BNS counterparts (e.g., BNS Section 101, BNS Section 303), slowing down casework and legal filings.
3. **Information Overload and Unstructured Data**: Law offices, courts, and corporate legal departments deal with thousands of pages of unstructured case files, contracts, and judgements in PDF formats. Manually indexing, categorizing, and summarizing these files is time-consuming, prone to human error, and lacks scalable mechanisms for topic discovery or anomaly detection.

Existing chat tools are general-purpose and frequently hallucinate legal sections or punishments, combining unrelated clauses or creating incorrect section pairings. This project addresses these gaps by creating a localized, grounded, domain-aware retrieval system and automated analysis pipeline.

---

## 4. Objectives
The primary objectives of the Smart Legal Assistance system are:
- **Democratize Legal Knowledge**: Provide citizens with a natural-language search tool that translates complex statutory clauses into plain English definitions, outline penalties, and describe filing procedures.
- **Provide Section-Level Accuracy**: Ensure that statutory references, punishment terms, and fine structures are grounded in source PDFs via hybrid retrieval, preventing the merging or confusion of distinct clauses.
- **Synthesize IPC-BNS Interoperability**: Maintain a dual-metadata schema mapping IPC and BNS sections, enabling immediate lookup of equivalent sections from both historical and updated criminal acts.
- **Automate Corpus Exploration**: Build an end-to-end unsupervised pipeline that ingests arbitrary legal PDFs, groups them into logical thematic clusters, extracts keywords, maps topics, and highlights anomalies without human-supervised labels.
- **Enable Accessibility and Portability**: Provide high-speed text-to-speech audio feedback for auditory users, compile case summaries into downloadable PDF reports, and packaging the entire application in a containerized environment (Docker Compose) for easy deployment.

---

## 5. Scope
The scope of the project encompasses:
- **Target Users**: General citizens seeking procedural clarity, law students needing reference resources, and legal advocates searching case files.
- **Legal Boundaries**: Centered on criminal law acts (primarily IPC and BNS structures). The system allows arbitrary custom PDFs to be uploaded, making the retrieval framework generalizable to contracts, corporate bylaws, and court transcripts.
- **Support Limitations**: The system is designed for educational guidance and basic documentation assistance. It does not execute legal filings, negotiate settlements, or replace a licensed attorney.
- **Hardware Profile**: The system is designed to run locally on consumer hardware. Large Language Models and embedding encoders are selected for low memory footprint and high CPU performance, minimizing cloud dependencies and safeguarding document privacy.

---

## 6. Features
The Smart Legal Assistance platform contains the following application features:

| Feature Area | Description | Implementation File(s) |
| :--- | :--- | :--- |
| **Interactive Dashboard** | View high-level metrics of the system: total registered users, uploaded documents, chat logs, and overall user feedback ratings. | [dashboard.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/ui/dashboard.py) |
| **User Authentication** | Secure registration and login using bcrypt password hashing, session management, and role/username tracking. | [login.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/auth/login.py), [register.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/auth/register.py) |
| **Hybrid RAG Chat** | Question-answering interface powered by FAISS semantic search, BM25 keyword matching, exact section lookups, and a Qwen LLM. | [chatbot.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/ui/chatbot.py), [enhanced_rag_engine.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/src/enhanced_rag_engine.py) |
| **Local Text-to-Speech** | Preprocesses assistant answers (expanding IPC/BNS terms, removing markdown, formatting currencies) and synthesizes audio via Piper TTS. | [tts.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/src/tts.py) |
| **Dynamic PDF Ingestion** | Upload legal PDFs directly through the UI. Extracts text or tables, runs a metadata validator, generates embeddings, and updates the active vector index. | [upload.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/ui/upload.py), [document_ingestion.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/src/document_ingestion.py) |
| **Document Explorer** | Browse processed document chunks, view source paths, metadata fields, and explore discovered unsupervised topic groups. | [explorer.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/ui/explorer.py) |
| **Feedback System** | Users can rate chat answers (1-5 stars) and submit detailed comments. The database logs this to help evaluate and fine-tune system performance. | [feedback.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/ui/feedback.py) |
| **Settings Panel** | View database details, configure TTS parameters, clean caches, or reset session states. | [settings.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/ui/settings.py) |

---

## 7. System Architecture
The application is structured as a modular, three-tier architecture: the Client Tier (Streamlit frontend), the Application Logic Tier (hybrid retrieval, NLP pipeline, and machine learning components), and the Data Tier (MySQL database, FAISS index, and file caches). 

Below is the conceptual architecture diagram of the platform:

```
+---------------------------------------------------------------------------------+
|                                 CLIENT TIER                                     |
|                      Streamlit Frontend User Interface                          |
|   +---------------+  +---------------+  +--------------+                                |
|   |   Dashboard   |  | RAG Chat UI   |  | PDF Ingester |                                |
|   +---------------+  +---------------+  +--------------+                                |
+------------------------------|--------------------------------------------------+
                               | Request / Session State
                               v
+---------------------------------------------------------------------------------+
|                            APPLICATION LOGIC TIER                               |
|   +-------------------------------------------------------------------------+   |
|   |                         Intent Classifier Routing                       |   |
|   +------------------------------------+------------------------------------+   |
|                                        |
|                     +------------------+------------------+
|                     |                                     |
|                     v [Interactive Chat Flow]             v [Offline Clustering Pipeline]
|   +----------------------------------+  +----------------------------------+    |
|   |         RAG Search Engine        |  |  Unsupervised Discovery Pipeline |    |
|   |  - Exact IPC/BNS Section Lookup  |  |  - PyMuPDF Text Ingestor         |    |
|   |  - Sentence-Transformer Embedder |  |  - PyTesseract OCR Fallback      |    |
|   |  - BM25 Keyword Search           |  |  - SpaCy Lemmatizer / Stopwords  |    |
|   |  - Cross-Encoder Reranker        |  |  - Sliding-Window Chunker        |    |
|   |  - Local Ollama LLM Generator    |  |  - K-Means Auto-K Silhouette     |    |
|   |  - Grounding Hallucination Guard |  |  - BERTopic Modeling Engine      |    |
|   |  - Piper TTS Voice Engine        |  |  - Matplotlib & Plotly Dashboard |    |
|   +------------------+---------------+  +-----------------+----------------+    |
+----------------------|------------------------------------|---------------------+
                       | CRUD Operations /                  | Save Artifacts /
                       | Vector Index Reads                 | Read Clusters
                       v                                    v
+---------------------------------------------------------------------------------+
|                                  DATA TIER                                      |
|   +-----------------------+  +-----------------------+  +-------------------+   |
|   |    MySQL Database     |  |      FAISS Index      |  | Audio WAV Cache   |   |
|   |  - Users, Chat Logs,  |  |  - FlatL2 Vectors     |  |  - Temp TTS Files |   |
|   |    Feedback, Audits   |  |  - Metadata Pickle    |  |    (Max 50MB)     |   |
+---------------------------------------------------------------------------------+
```

---

## 8. Workflow
The application operates on three primary workflows.

### A. Document Ingestion Workflow
When a user uploads a PDF document via the Streamlit frontend, the following sequential processing steps occur:
1. **Extraction Strategy Selection**: `DocumentIngestion` receives the PDF path. It passes it to `TableExtractor` to check for structured tables.
2. **Structured Table Extraction**: `TableExtractor` uses `pdfplumber` to search for grids. If headers match key columns (e.g. "IPC Section", "Offence", "Punishment"), columns are mapped to schemas. Wrapped rows are merged.
3. **Fallback Extraction**: If no tables are found, `DocumentIngestion` calls `PDFLoader` and `DocumentChunker` to extract raw text and split it into sliding window chunks of 512 characters with a 64-character overlap, tracking source pages.
4. **Validation and Enrichment**: Each chunk is analyzed by `MetadataValidator`. Missing sections (IPC/BNS), crime labels, and punishments are filled in using regex and keyword lists. The validator builds a single structured text payload: `complete_text`. Chunks missing required fields are discarded.
5. **Vector Database Update**: `EmbeddingGenerator` embeds the `complete_text` strings into 384-dimensional dense vectors using the `all-MiniLM-L6-v2` transformer. These vectors are added to the FAISS FlatL2 index, and the index is serialized to `models/faiss_index.bin`.

### B. Chatbot Query Resolution Workflow
When a user asks a question (e.g., *"What is the punishment for cheating under BNS?"*), the query is resolved through the following steps:
1. **Session Memory & Context Retrieval**: `EnhancedChatbot` intercepts the query and reads the recent conversation logs from `MemoryManager`.
2. **Query Rewriting**: If the query is a follow-up, `QueryRewriter` uses conversation state to expand pronoun references (e.g., *"What is its penalty?"* becomes *"What is the penalty for cheating?"*).
3. **Intent Classification**: `IntentClassifier` runs a regex engine matching the query against intent groups (e.g., punishment, section lookup, scenario, greeting).
4. **Hybrid Retrieval**: `Retriever` executes three searches:
   - **Exact Code Search**: Parses sections (e.g., "403") and matches them against IPC and BNS indexes.
   - **Semantic Search**: Generates a query embedding and runs a FAISS FlatL2 similarity lookup.
   - **Keyword Search**: Performs a BM25 lookup.
5. **Reranking**: Chunks retrieved across exact, semantic, and keyword searches are combined. A Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`) reranks them based on direct semantic relevance to the query. The top 5 chunks are selected.
6. **Context Formulation**: `ContextBuilder` merges overlapping chunks and creates a formatted prompt context.
7. **Response Generation**: The prompt is sent to the local Ollama LLM, which is structured to return a consistent markdown output containing Definition, IPC Section, BNS Section, Punishment, and Additional Notes.
8. **Hallucination Guard Verification**: `HallucinationGuard` checks the response. If the confidence score is low, it returns a safe fallback message. Otherwise, it checks for keyword overlap (minimum 20% overlap required) between the answer and the retrieved context. If valid, the response is displayed, and the sources are appended.
9. **Text-To-Speech Synthesis**: If enabled, the user can click "Listen". `TTSManager` cleans markdown elements, expands abbreviations, converts currency, generates a cached `.wav` file, and plays the audio.

### C. Unsupervised Learning Workflow
The offline analysis pipeline processes a collection of PDF files through a 10-phase sequence:

```
[Phase 1: Ingestion]  --> Reads files and falls back to OCR if scanned (ocr_handler.py)
         |
         v
[Phase 2: Preprocessing] -> Lemmatization, case folding, legal citation stripping (preprocessing.py)
         |
         v
[Phase 3: Chunking]   --> splits into sliding 512-character window blocks (chunking.py)
         |
         v
[Phase 4: Embeddings] --> Computes dense 384-dim semantic embeddings (feature_extraction.py)
         |
         v
[Phase 5: Clustering] --> Automatically finds optimal K via silhouette score (clustering.py)
         |
         v
[Phase 6: Topics]     --> Fits BERTopic, assigns document probabilities (topic_modeling.py)
         |
         v
[Phase 7: Discovery]  --> Compiles insights, flags outliers, computes coherence (knowledge_discovery.py)
         |
         v
[Phase 8: Storage]    --> Saves embeddings (.npy) and insights (.json) (storage.py)
         |
         v
[Phase 9: Visuals]    --> PCA scatter plots, heatmaps, and dashboard figures (visualization.py)
         |
         v
[Phase 10: Report]    --> Aggregates key findings and metadata summaries (unsupervised_app.py)
```

---

## 9. Technology Stack
The Smart Legal Assistance application is built on the following technologies:

### Frontend Tier
- **Streamlit (v1.35.0+)**: Used to build the interactive web frontend, supporting multi-page routing, session states, file uploaders, and native chat widgets.

### Application & Processing Tier
- **Python (v3.12)**: Core programming language.
- **SQLAlchemy (ORM)**: Database toolkit and Object-Relational Mapper to connect python classes to MySQL schemas.
- **Sentence-Transformers**: Generates dense semantic embeddings (`all-MiniLM-L6-v2` model) and reranks results (`cross-encoder/ms-marco-MiniLM-L-6-v2` model).
- **FAISS (Facebook AI Similarity Search)**: Efficient similarity search library for dense vectors.
- **pdfplumber**: Extract tables and grids from PDFs.
- **PyMuPDF (fitz)**: Fast text extraction from PDF files.
- **PyTesseract & Tesseract OCR Engine**: OCR library used as a fallback to extract text from scanned image-only PDFs.
- **SpaCy (`en_core_web_sm`)**: Used for lemmatization, sentence boundary detection, and entity extraction.
- **BERTopic**: Unsupervised topic modeling framework that uses HDBSCAN and c-TF-IDF.
- **Piper TTS**: Fast, local text-to-speech engine running as a subprocess to synthesize WAV audio.
- **Ollama**: Local containerized LLM orchestrator running the Qwen model.

### Database & Operations Tier
- **MySQL**: Relational database engine.
- **Docker & Docker Compose**: Containerization tool that bundles the Streamlit application and the MySQL database into a single multi-container environment.

---

## 10. Folder Structure
The project folder structure is organized as follows:

```
Smart_Legal_Assistance/
├── .gitignore                   # Version control file filters
├── Dockerfile                   # Streamlit app container definition
├── docker-compose.yml           # Multi-container orchestration (Streamlit & MySQL)
├── requirements.txt             # Pinned python dependencies
├── streamlit_app.py             # Main entry point for the Streamlit UI
├── unsupervised_app.py          # Entry point for the 10-phase unsupervised pipeline
├── run_assistant.py             # CLI runner script for RAG evaluation
├── auth/                        # User authentication and security
│   ├── login.py                 # Renders Login page and verifies credentials
│   ├── register.py              # Renders Registration page and inserts records
│   ├── security.py              # Password hashing helper (bcrypt)
│   └── session_manager.py       # Tracks active user IDs and roles in Streamlit
├── config/                      # Application settings
│   ├── .env.example             # Template file for database passwords and paths
│   ├── config.py                # Environment parser and constants
│   └── tts_config.json          # Piper TTS settings (voice, speed, read sources)
├── database/                    # Persistence layer config
│   ├── connection.py            # Sets up SQLAlchemy Engine and SessionLocal
│   └── models.py                # Maps User, Document, ChatHistory, and Feedback schemas
├── docs/                        # Project documentation and reports
│   ├── COMPLETION_SUMMARY.md    # Summary of unsupervised completion
│   ├── PROJECT_DOCUMENTATION.md # THIS FILE
│   └── UNSUPERVISED_LEARNING.md # Implementation guide for clustering modules
├── generated_audio/             # Local cache directory for Piper WAV files
├── legal_pdfs/                  # Input folder for default PDFs on startup
├── logs/                        # Application logs
│   ├── app.log                  # Merged runtime logs
│   └── conversation_history.json# Cached session history
├── models/                      # FAISS files
│   ├── faiss_index.bin          # Indexed vector array
│   └── metadata.pkl             # Serialized metadata structures
├── piper/                       # Pre-built Piper TTS executable binaries
│   ├── piper.exe                # Executable binary
│   └── models/                  # ONNX voice model assets
├── processed_data/              # Outputs from unsupervised app runs
│   ├── embeddings.npy           # Binary dense vectors
│   ├── cluster_results.json     # Cluster indices and silhouette scores
│   ├── topic_results.json       # BERTopic probabilities and keywords
│   ├── processed_chunks.json    # Preprocessed and tokenized text chunks
│   ├── knowledge_discovery.json # Discovered patterns and anomalies
│   └── *.png                    # Dashboards and charts
├── src/                         # Backend source modules
│   ├── answer_generator.py      # Basic response generator
│   ├── chatbot.py               # Basic chat orchestrator
│   ├── chunker.py               # Document splitter
│   ├── chunking.py              # Unsupervised chunking wrapper
│   ├── clustering.py            # K-Means clustering and metrics
│   ├── confidence_estimator.py  # Calculates High/Medium/Low confidence
│   ├── context_builder.py       # Cleans and formats context for LLMs
│   ├── context_manager.py       # Context tracker
│   ├── context_resolver.py      # Context resolver
│   ├── document_ingestion.py    # Main pipeline to process PDFs
│   ├── embedding_generator.py   # Embedding generator
│   ├── enhanced_chatbot.py      # Chat wrapper (Query rewrites, memory checks)
│   ├── enhanced_rag_engine.py   # RAG pipeline manager
│   ├── entity_tracker.py        # Tracks entities
│   ├── evaluation.py            # RAG validation script
│   ├── feature_extraction.py    # SentenceTransformers embeddings module
│   ├── followup_resolver.py     # Follow-up resolver
│   ├── hallucination_guard.py   # Ensures responses are grounded in context
│   ├── intent_classifier.py     # Greedy regex query classifier
│   ├── knowledge_builder.py     # Generates JSON database from table extractions
│   ├── knowledge_discovery.py   # Summarizer, anomaly finder, insight generator
│   ├── legal_intent_mapper.py   # Intent mapper
│   ├── legal_term_mapper.py     # Maps terms
│   ├── memory.py                # History management
│   ├── memory_manager.py        # State tracker
│   ├── metadata_validator.py    # Metadata validator and enricher
│   ├── ocr_handler.py           # Tesseract wrapper
│   ├── ollama_generator.py      # LLM handler
│   ├── pdf_extractor.py         # PDF text extractor
│   ├── pdf_loader.py            # PDF loader
│   ├── preprocessing.py         # SpaCy text cleaner
│   ├── query_rewriter.py        # Contextual query expander
│   ├── rag_engine.py            # Original RAG processor
│   ├── response_templates.py    # Predefined responses
│   ├── retriever.py             # Hybrid retriever (FAISS + BM25 + Section)
│   ├── simplifier.py            # Response simplifier
│   ├── source_formatter.py      # Formatter
│   ├── storage.py               # Serializes data
│   ├── summarizer.py            # Document summarizer
│   ├── table_extractor.py       # pdfplumber table parser
│   ├── topic_modeling.py        # BERTopic manager
│   ├── tts.py                   # Piper TTS controller
│   ├── vector_store.py          # FAISS index wrapper
│   └── visualization.py         # Matplotlib dashboard generator
├── tests/                       # Unit and integration test suites
│   ├── conftest.py              # Pytest configuration and database fixtures
│   ├── test_answer_generation.py# Verifies answer formatting
│   ├── test_auth.py             # Tests login and registration
│   ├── test_chatbot.py          # Tests the chatbot interface
│   ├── test_context.py          # Tests context builder
│   ├── test_database.py         # Tests database connection and models
│   ├── test_feedback.py         # Tests feedback logging
│   ├── test_intent.py           # Tests intent classification
│   ├── test_query_rewriter.py   # Tests query rewriter
│   ├── test_rag.py              # Tests RAG pipeline
│   ├── test_retrieval.py        # Tests hybrid search accuracy
│   ├── test_streamlit.py        # Tests Streamlit page structure
│   ├── test_tts.py              # Tests text normalization and Piper TTS
│   └── test_upload.py           # Tests file uploading
└── ui/                          # Streamlit UI page components
    ├── chatbot.py               # Chatbot UI page
    ├── dashboard.py             # Application metrics UI page
    ├── explorer.py              # Document chunks viewer UI page
    ├── feedback.py              # Review submission UI page
    ├── settings.py              # Configuration editor UI page
    └── upload.py                # PDF upload UI page
```

---

## 11. Module Description

### Data Collection
The entry point for processing input files is [document_ingestion.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/src/document_ingestion.py). 
This module coordinates the ingestion pipeline:
- **Default Document Loading**: Automatically processes PDFs in the `legal_pdfs/` directory on application startup if the FAISS index is empty.
- **Routing**: Sends incoming files to `TableExtractor` to search for structured data. If no tables are found, it falls back to a page-by-page text loader and sliding window chunker.
- **Pipeline Orchestration**: Collects extracted chunks, runs them through the `MetadataValidator`, generates embeddings, adds them to the vector store, and validates the index.

### PDF Extraction
Text extraction is handled by two main modules:
- [pdf_extractor.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/src/pdf_extractor.py): Extracts text layout structures using `PyMuPDF` (fitz).
- [ocr_handler.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/src/ocr_handler.py): Uses `pytesseract` to scan image-only or low-text PDFs. It converts pages to images, preprocesses them (improving contrast and brightness), detects characters, and returns text along with extraction confidence scores.

### Text Preprocessing
Text cleaning and normalization are managed by [preprocessing.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/src/preprocessing.py):
- **SpaCy Integration**: Uses the `en_core_web_sm` model to tokenize texts and run lemmatization, reducing words to their base form (e.g., *cheated* becomes *cheat*).
- **Domain-Specific Cleaning**: Strips out legal case citations (such as *"1995 AIR 1234"* or *"Smith v. State"*), legal footnotes, page numbers, and custom legal stop words (e.g., *"court"*, *"petitioner"*) that do not contribute to semantic search context.
- **Feature Extraction**: Extracts Named Entities (such as PERSON, ORG, and GPE) to enrich metadata tags.

### Embedding Generation
Semantic vector generation is handled by [embedding_generator.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/src/embedding_generator.py) and [feature_extraction.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/src/feature_extraction.py):
- **dense vector generation**: Wraps `SentenceTransformer` using the `all-MiniLM-L6-v2` model.
- **Batch Processing**: Encodes documents in batches of 32 chunks. This maximizes GPU or multi-core CPU usage while keeping memory overhead low.
- **Array Output**: Returns 384-dimensional dense float arrays mapping the semantic meanings of chunks.

### Vector Database
The vector storage engine is implemented in [vector_store.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/src/vector_store.py):
- **FAISS Index FlatL2**: Uses L2 Euclidean distance arrays to run high-speed similarity searches on dense vectors.
- **Dual Schema Support**: Handles both historic schemas (which use a single `section_number` field) and modern schemas (which split sections into separate `ipc_section` and `bns_section` fields).
- **Exact Matches**: Indexes metadata keys to allow fast, direct matches on IPC sections, BNS sections, or specific crime names without semantic overhead.
- **Index Health Checks**: Verifies that the number of vectors matches the metadata database size, and checks the ratio of mapped sections.

### Query Classification
User queries are routed through [intent_classifier.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/src/intent_classifier.py):
- **Intent Detection**: Analyzes query text using regex mapping rules to identify the user's focus: greeting, section lookup, scenario analysis, punishment details, comparisons, or follow-ups.
- **Dynamic Classification**: Returns a structured `IntentResult` container that carries both the primary intent category and a sub-intent descriptor. This is used by the retriever to adjust scoring weights.

### RAG Pipeline
Retrieval and context ranking are managed by [enhanced_rag_engine.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/src/enhanced_rag_engine.py) and [retriever.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/src/retriever.py):
- **Hybrid Scoring**: Combines FAISS semantic search scores (35% weight), BM25 keyword search scores (25% weight), and exact-match section matches (100% boost) into a single metric.
- **Intent-Based Boosting**: Boosts similarity scores (by a factor of 1.2) for semantic search if the intent is "definition" or "scenario", and for keyword search if the intent is "punishment" or "section_lookup".
- **Cross-Encoder Reranking**: Uses `ms-marco-MiniLM-L-6-v2` to predict direct relevance scores for the top 20 retrieved chunks. It re-sorts them and passes the top 5 to the context builder.
- **Grounding Guard**: Uses `HallucinationGuard` to calculate keyword overlap between the LLM's response and the retrieved source material, filtering out ungrounded information.

### LLM Response Generation
Text generation is handled by [ollama_generator.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/src/ollama_generator.py) and [enhanced_chatbot.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/src/enhanced_chatbot.py):
- **Ollama Integration**: Connects to the local Ollama daemon to stream responses chunk-by-chunk.
- **Strict Layout Control**: Prompts the model to return structured markdown with standardized headers: `### Definition`, `### Applicable IPC Section`, `### Applicable BNS Section`, `### Punishment`, `### Additional Notes`, and `### Sources`.
- **Memory Management**: Keeps a history window of the last 6 conversation turns, enabling follow-up questions (such as *"What is the punishment under the new act?"*) by rewriting queries with context.

### Streamlit Frontend
The user interface is built using Streamlit modules in the [ui/](file:///d:/Skylena/SLA/Smart_Legal_Assistance/ui) directory:
- [dashboard.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/ui/dashboard.py): Displays key metrics (total users, documents, and ratings).
- [upload.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/ui/upload.py): Provides a file drag-and-drop zone to upload PDFs.
- [chatbot.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/ui/chatbot.py): Renders the chat window, manages session messages, logs chat history to the database, and hosts the Text-to-Speech playback controls.
- [explorer.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/ui/explorer.py): Lets users inspect vector chunks, page references, and metadata attributes.
- [feedback.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/ui/feedback.py): Renders rating options to save user feedback.
- [settings.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/ui/settings.py): Configures audio generation speed and clears session states.

---

## 12. Dataset Description
The application reads and extracts data from unstructured legal documents:
- **Default Documents**: The system initializes using a legal dataset placed in `legal_pdfs/`. This includes tabular PDF guides that map IPC offenses directly to BNS sections, detailed criminal descriptions, and statutory punishments.
- **Document Chunk Schema**: Once processed, every text chunk is validated against a metadata schema. The schema requires the following fields:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `chunk_id` | Integer / String | Unique identifier for the chunk. |
| `source_document` | String | Filename of the source PDF. |
| `ipc_section` | String | Normalized IPC section number (e.g., *"378"*). |
| `bns_section` | String | Normalized BNS section number (e.g., *"303"*). |
| `crime_name` | String | Standardized title of the offense (e.g., *"Theft"*). |
| `description` | String | Detailed statutory definition or description. |
| `punishment` | String | Prescribed statutory punishment (imprisonment, fine, or both). |
| `related_sections` | String | Other relevant legal sections mentioned in the text. |
| `page_number` | Integer | Page number where the text was found in the source PDF. |
| `keywords` | Array of Strings | Key terms extracted for indexing and filtering. |
| `aliases` | Array of Strings | Synonyms or alternate names for the offense. |
| `complete_text` | String | Formatted text block containing all fields, used for embedding generation. |

---

## 13. Installation Guide

### Prerequisites
1. **Python 3.12+**: Ensure Python is installed and added to your system path.
2. **Tesseract OCR Engine**:
   - **Windows**: Download the installer from the [UB-Mannheim Tesseract Repo](https://github.com/UB-Mannheim/tesseract/wiki). Install it to `C:\Program Files\Tesseract-OCR` and add that path to your system's Environment Variables.
   - **Linux/Ubuntu**: Run `sudo apt-get install tesseract-ocr libtesseract-dev`.
3. **Ollama**: Download and install Ollama from [ollama.com](https://ollama.com). Start the service and run `ollama pull qwen3:0.6b` (or your preferred Qwen model version).

### Step-by-Step Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd Smart_Legal_Assistance
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the SpaCy NLP Model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

---

## 14. Project Setup

### Option A: Running with Docker Compose (Recommended)
This approach sets up both the Streamlit application and the MySQL database in containers, linking them automatically.

1. **Verify Docker Status**: Ensure Docker Desktop is running.
2. **Launch Services**:
   ```bash
   docker-compose up --build
   ```
3. **Access the Application**: Open your browser and go to `http://localhost:8501`.

### Option B: Local Running (Without Docker)
1. **Set Up MySQL Database**: Set up a local MySQL instance and create a database named `smart_legal_assistance`.
2. **Configure Environment Variables**:
   - Copy `config/.env.example` to `config/.env`.
   - Edit `config/.env` and update the database credentials:
     ```env
     DB_USER=your_mysql_user
     DB_PASSWORD=your_mysql_password
     DB_HOST=127.0.0.1
     DB_PORT=3306
     DB_NAME=smart_legal_assistance
     ```
3. **Configure Piper TTS**:
   - Ensure the executable is at `piper/piper.exe`.
   - Download the ONNX voice model `en_US-lessac-medium.onnx` and its `.json` config, placing them in `piper/models/`.
4. **Run Streamlit**:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 15. Implementation Details
- **Database Model Mapping**: Managed using SQLAlchemy ORM in [models.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/database/models.py). The schema defines relationships between `User`, `ChatHistory`, and `Feedback`. For audit security, the `AuditLog` table records user logins and uploads.
- **Modular Retrieval Design**: The retriever separates keyword extraction, exact database queries, and semantic searches. This modular structure allows developers to swap components (e.g., replacing FAISS with PgVector) without rewriting search workflows.
- **Piper TTS Caching**: To minimize speech synthesis delays, `TTSManager` hashes the input text and voice settings. If the matching `.wav` file is already in the cache, the system skips generation and plays the cached file. A cleanup routine runs after each generation, removing oldest files first if the cache directory exceeds 50MB.

---

## 16. Testing and Validation
The project includes a suite of unit and integration tests located in the `tests/` directory.

### Running Automated Tests
Run the test suite using `pytest`:
```bash
pytest tests/
```

### Key Test Suites
- [test_intent.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/tests/test_intent.py): Validates that the intent classifier correctly routes greetings, scenario questions, and section lookups.
- [test_retrieval.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/tests/test_retrieval.py): Tests the hybrid search, checking that exact matches on IPC or BNS section numbers return correct documents.
- [test_tts.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/tests/test_tts.py): Verifies text cleaning rules, such as replacing currency symbols and expanding abbreviations, and checks Piper subprocess calls.
- [test_database.py](file:///d:/Skylena/SLA/Smart_Legal_Assistance/tests/test_database.py): Tests database operations, ensuring user credentials, chat histories, and feedback submissions save correctly.

---

## 17. Challenges Faced
Developing this application presented several key challenges:
1. **Merging Table Cells in PDF Ingestion**: Legal PDFs often split long descriptions or multi-line punishments across several rows. A naive PDF parser would treat these as separate chunks, losing context. This was solved by updating `TableExtractor` to check the section columns; if they are empty, it appends the text to the previous row.
2. **LLM Hallucinations and Format Slippage**: Small local models (like `qwen3:0.6b`) sometimes deviate from formatting instructions. To address this, we added the `HallucinationGuard` layer to enforce structured formatting, run keyword overlap checks, and fall back to a standard warning message if the output is ungrounded.
3. **MySQL Connection Delays in Docker**: In containerized environments, the Streamlit app can start before the MySQL database finishes initializing, causing connection errors. This was resolved by adding a retry loop in Streamlit's startup configuration and configuring database checks.
4. **TTS Subprocess Optimization**: Running the Piper executable as a subprocess for every chat message can cause delays. We addressed this by implementing MD5-hashed caching for audio files, significantly speeding up playback for duplicate queries.

---

## 18. Future Enhancements
Future updates to the Smart Legal Assistance application could include:
- **GPU Acceleration**: Add CUDA support to speed up SentenceTransformers embedding generation and reranking.
- **Vector Store Scaling**: Transition from local FAISS files to a centralized database (such as pgvector or Milvus) to support larger document sets.
- **Multilingual Support**: Add translation layers to allow users to ask questions and receive answers in regional Indian languages.
- **Automated Case Drafting**: Expand the assistant's capabilities to generate basic legal drafts, such as draft FIR complaints or tenant agreements, using retrieved document templates.
- **Enhanced OCR**: Integrate advanced OCR tools (like EasyOCR or DocTR) to improve text extraction from low-resolution or handwritten document scans.

---

## 19. Conclusion
The **Smart Legal Assistance** application provides a secure, grounded, and accessible solution for navigating legal texts. By combining structured table extraction, hybrid search, cross-encoder reranking, and local LLMs, the platform offers accurate, section-level answers while avoiding the common hallucination issues of general chat tools. 

Its unsupervised learning pipeline allows for automated, unsupervised exploration of large legal document sets, helping users identify hidden topics, anomalies, and insights. This modular, Docker-ready implementation demonstrates how localized AI technologies can help make legal information more transparent and accessible.

---

## 20. References
1. **SentenceTransformers**: Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *arXiv preprint arXiv:1908.10084*.
2. **FAISS (Facebook AI Similarity Search)**: Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535-547.
3. **BERTopic**: Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv preprint arXiv:2203.05794*.
4. **pdfplumber**: pdfplumber contributors. (2024). Plumb a PDF for detailed info on characters, lines, and tables. *GitHub Repository*.
5. **Piper TTS**: Rhasspy Voice Assistant Project. (2023). A fast, local neural text-to-speech system. *GitHub Repository*.
6. **Ollama**: Ollama Authors. (2024). Run Llama 3, Mistral, and other large language models locally. *Ollama Project*.
