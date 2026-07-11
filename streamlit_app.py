import streamlit as st
import os
from dotenv import load_dotenv

# Load env variables early
load_dotenv(os.path.join(os.path.dirname(__file__), 'config', '.env'))

# Ensure DB tables exist
from database.connection import engine, Base
Base.metadata.create_all(bind=engine)

from auth.session_manager import init_session, is_authenticated, logout_user
from auth.login import render_login
from auth.register import render_register
from ui.dashboard import render_dashboard
from ui.upload import render_upload
from ui.chatbot import render_chatbot
from ui.explorer import render_explorer
from ui.feedback import render_feedback
from ui.settings import render_settings

# RAG imports
from src.vector_store import VectorStore
from src.document_ingestion import DocumentIngestion
from src.retriever import Retriever
from src.enhanced_rag_engine import EnhancedRAGEngine
from src.enhanced_chatbot import EnhancedChatbot

st.set_page_config(page_title="Smart Legal Assistance", page_icon="⚖️", layout="wide")

@st.cache_resource
def init_rag_system(version=7):
    st.sidebar.info("Initializing RAG models... (This may take a moment)")
    vs = VectorStore(
        index_path="models/faiss_index.bin",
        metadata_path="models/metadata.pkl"
    )
    
    # Try to load existing index
    try:
        vs.load_index()
    except Exception as e:
        import logging
        logger = logging.getLogger("rag_engine")
        logger.exception(e)
        print("No existing FAISS index found or an error occurred. A new one will be built.")

    ingestion = DocumentIngestion(vs)
    
    # Process default documents on startup
    ingestion.load_default_documents()

    retriever = Retriever(vs, ingestion.embedder)
    rag_engine = EnhancedRAGEngine(retriever, model_name="qwen3:0.6b")
    chatbot = EnhancedChatbot(rag_engine)
    
    return ingestion, chatbot

def main():
    init_session()
    
    # Display advocate avatar at the top of the sidebar
    avatar_path = os.path.join(os.path.dirname(__file__), "assets", "advocate_avatar.png")
    if os.path.exists(avatar_path):
        st.sidebar.image(avatar_path, use_container_width=True)
    
    if not is_authenticated():
        auth_mode = st.sidebar.radio("Authentication", ["Login", "Register"])
        if auth_mode == "Login":
            render_login()
        else:
            render_register()
    else:
        st.sidebar.title(f"Welcome, {st.session_state['username']}")
        
        ingestion, chatbot = init_rag_system()
        
        pages = {
            "Dashboard": render_dashboard,
            "Upload PDF": lambda: render_upload(ingestion),
            "Chatbot": lambda: render_chatbot(chatbot),
            "Document Explorer": render_explorer,
            "Feedback": render_feedback,
            "Settings": render_settings
        }
        
        selection = st.sidebar.radio("Navigate", list(pages.keys()))
        
        # Render selected page
        pages[selection]()
        
        st.sidebar.divider()
        if st.sidebar.button("Logout"):
            logout_user()
            st.rerun()

if __name__ == "__main__":
    main()
