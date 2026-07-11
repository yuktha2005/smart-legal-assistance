import streamlit as st
import pandas as pd
from database.connection import get_db
from database.models import Document

def render_explorer():
    st.title("Document Explorer")
    st.write("View details of uploaded documents, extracted topics, and clusters.")
    
    db_generator = get_db()
    db = next(db_generator)
    
    try:
        documents = db.query(Document).all()
        
        if not documents:
            st.info("No documents found. Please upload a PDF first.")
            return
            
        doc_data = [{
            "ID": doc.id,
            "Filename": doc.filename,
            "Upload Date": doc.upload_date,
            "Pages": doc.pages,
            "Chunks": doc.chunk_count
        } for doc in documents]
        
        df = pd.DataFrame(doc_data)
        st.dataframe(df, use_container_width=True)
        
        st.subheader("Topic Clusters")
        st.write("Select a document to explore its topics.")
        
        selected_doc = st.selectbox("Select Document", df['Filename'].tolist())
        
        if selected_doc:
            # Placeholder for topic retrieval from Unsupervised Learning module
            # topics = get_topics_for_document(selected_doc)
            st.write(f"Showing clusters for: **{selected_doc}**")
            # Mock topics
            st.json({
                "Cluster 0": ["contract", "agreement", "terms"],
                "Cluster 1": ["liability", "damages", "breach"],
                "Cluster 2": ["court", "jurisdiction", "appeal"]
            })
            
    finally:
        db_generator.close()
