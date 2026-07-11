# Smart Legal Assistance (Phase 3: Production Release)

The Phase 3 release transforms the **SMART LEGAL ASSISTANCE** backend into a fully functional, client-ready web application using Streamlit, MySQL, and Docker.

## Architecture & Tech Stack
- **Frontend**: Streamlit (Dashboard, Chatbot, Document Explorer)
- **Backend**: Python 3.12+ (existing RAG, NLP, and extraction pipelines)
- **Database**: MySQL (managed via SQLAlchemy ORM)
- **Vector Store**: FAISS
- **Authentication**: bcrypt password hashing
- **Deployment**: Docker & Docker Compose

## Folder Structure
```
Smart_Legal_Assistance/
  auth/           # Registration, login, and session state
  config/         # Environment variables (.env)
  database/       # SQLAlchemy models and connection
  reports/        # PDF ReportLab generation
  src/            # Core RAG, VectorStore, NLP processing
  tests/          # Pytest suites
  ui/             # Streamlit page components
  streamlit_app.py # Main UI Entry Point
  Dockerfile
  docker-compose.yml
  requirements.txt
```

## Setup & Installation

### Option 1: Docker (Recommended)
The entire application (Streamlit + MySQL) is containerized.
1. Make sure Docker Desktop is running.
2. Run the application:
```bash
docker-compose up --build
```
3. Open your browser and navigate to `http://localhost:8501`.

### Option 2: Local Development (Without Docker)
1. Install Python dependencies:
```bash
pip install -r requirements.txt
```
2. Set up MySQL locally and create a database named `smart_legal_assistance`.
3. Rename `config/.env.example` to `config/.env` and update the `DB_PASSWORD` and `DB_USER`.
4. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## Features
- **Dashboard**: View high-level metrics of users, documents, and chat sessions.
- **Upload PDF**: Process new legal documents dynamically (extracts text, chunks, embeddings, and updates FAISS).
- **Chatbot**: ChatGPT-like interface with conversation memory, source citations, and confidence scores.
- **Document Explorer**: Browse uploaded documents and topic clusters.
- **Feedback Module**: Provide ratings and comments for system improvement.
- **Settings**: Manage data and clear conversation histories.

## Testing
Run the test suite using `pytest`:
```bash
pytest tests/
```

## Troubleshooting
- **Database Connection Refused**: Ensure the MySQL container is healthy. The Streamlit container has a `depends_on` rule, but MySQL might take a few extra seconds to initialize.
- **Missing FAISS Index**: Make sure to upload at least one PDF through the UI to generate the initial FAISS index if running on a fresh volume.
