import os
import streamlit as st
import shutil
from database.connection import get_db
from database.models import Document

def render_upload(ingestion=None):
    st.title("Upload Legal Documents")
    st.write("Upload PDFs to extract text, generate embeddings, and update the FAISS vector database automatically.")
    
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if not ingestion:
            st.error("Document ingestion pipeline is not initialized.")
            return

        if uploaded_files:
            upload_dir = os.getenv("UPLOAD_FOLDER", "legal_pdfs/")
            os.makedirs(upload_dir, exist_ok=True)
            
            db_generator = get_db()
            db = next(db_generator)
            
            try:
                for file in uploaded_files:
                    file_path = os.path.join(upload_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    with st.spinner(f"Processing {file.name} (Extracting text, Chunking, Generating embeddings)..."):
                        num_chunks = ingestion.process_pdf(file_path)
                    
                    # Save to DB
                    # Estimate pages (for real logic, PyMuPDF len(doc) could be used. Here we assume 1 page per 2 chunks)
                    pages = max(1, num_chunks // 2)
                    
                    new_doc = Document(filename=file.name, pages=pages, chunk_count=num_chunks)
                    db.add(new_doc)
                
                db.commit()
                st.success("Knowledge updated successfully. The chatbot can now answer from these documents!")
                
            except Exception as e:
                db.rollback()
                st.error(f"An error occurred during processing: {e}")
            finally:
                db_generator.close()
        else:
            st.warning("Please upload at least one PDF file.")
