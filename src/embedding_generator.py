from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Generates dense vector embeddings for a list of strings.
        Returns a numpy array of shape (N, D).
        """
        if not texts:
            return np.array([])
            
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        # --- DEBUG LOGS (STEP 2) ---
        print("\n" + "="*50)
        print("STEP 2 : VERIFY EMBEDDINGS")
        print("="*50)
        print(f"Embedding shape: {embeddings.shape}")
        if len(texts) > 0:
            print("Preview 1st chunk:")
            print(f"{texts[0][:100]}...")
        print("="*50 + "\n")
        
        return embeddings
