import os
import uuid
from src.knowledge_builder import KnowledgeBuilder
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import VectorStore

class DocumentIngestion:
    def __init__(self, vector_store: VectorStore, model_name: str = 'all-MiniLM-L6-v2'):
        self.vector_store = vector_store
        self.builder = KnowledgeBuilder()
        self.embedder = EmbeddingGenerator(model_name)

    def process_pdf(self, file_path: str):
        """
        Extracts, structures, embeds, and appends the PDF to the FAISS index.
        """
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        
        # Try structured extraction first for ALL PDFs
        records = self.builder.build_knowledge_base(file_path)
        
        chunks = []
        if records:
            print(f"  -> TableExtractor found {len(records)} records")
            for r in records:
                chunks.append({
                    "chunk_id": len(chunks),
                    "source_document": filename,
                    "ipc_section": r.get('ipc_section', 'N/A'),
                    "bns_section": r.get('bns_section', 'N/A'),
                    "crime_name": r.get('crime', ''),
                    "description": r.get('description', ''),
                    "ingredients": r.get('ingredients', ''),
                    "punishment": r.get('punishment', ''),
                    "important_cases": r.get('important_cases', ''),
                    "related_sections": r.get('related_sections', 'N/A'),
                    "keywords": r.get('keywords', []),
                    "page_number": r.get('page_number', 1),
                    "text": r.get('description', '')
                })
        else:
            # Fallback to intelligent chunker
            print(f"  -> TableExtractor found 0 records, falling back to chunker")
            from src.pdf_loader import PDFLoader
            from src.chunking import DocumentChunker
            loader = PDFLoader()
            chunker = DocumentChunker()
            pages_text = loader.extract_text(file_path)
            if not pages_text:
                print(f"  -> Error: PDFLoader returned no text for {filename}")
                return 0
                
            # Process page by page to preserve exact page numbers
            for p in pages_text:
                p_chunks = chunker.chunk_document(p["text"], doc_id=filename, doc_source=filename)
                for pc in p_chunks:
                    pc["page_number"] = p["page_number"]
                    pc["chunk_id"] = len(chunks)
                    chunks.append(pc)
            
        if not chunks:
            return 0
            
        # Phase 4: Validate and enrich metadata
        from src.metadata_validator import MetadataValidator
        validator = MetadataValidator()
        
        validated_chunks = []
        for c in chunks:
            vc = validator.validate_and_enrich(c)
            if vc is not None:
                validated_chunks.append(vc)
                
        validator.print_report()

        if not validated_chunks:
            print(f"  -> Warning: No validated chunks found for {filename}")
            return 0

        # Embed complete_text (structured legal record) instead of raw text
        texts = [c["complete_text"] for c in validated_chunks]
        embeddings = self.embedder.generate_embeddings(texts)

        # Update FAISS and save
        self.vector_store.add_embeddings(embeddings, validated_chunks)
        self.vector_store.save_index()
        
        return len(validated_chunks)

    def load_default_documents(self, default_dir: str = "legal_pdfs"):
        """Loads default documents if the FAISS index is empty."""
        # Check if vector store already has data
        if self.vector_store.index is not None and self.vector_store.index.ntotal > 0:
            return
            
        if not os.path.exists(default_dir):
            return
            
        for file in os.listdir(default_dir):
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(default_dir, file)
                try:
                    self.process_pdf(file_path)
                except Exception as e:
                    print(f"Failed to load default doc {file}: {e}")
                    
        # Phase 5: Run validation automatically after ingestion
        self.vector_store.validate_index()
