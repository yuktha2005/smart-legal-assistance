import math
from collections import Counter
import re
from src.vector_store import VectorStore
from src.embedding_generator import EmbeddingGenerator


class SimpleBM25:
    def __init__(self, corpus: list[list[str]]):
        self.corpus_size = len(corpus)
        self.avgdl = sum(map(len, corpus)) / self.corpus_size if self.corpus_size else 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        
        nd = {}
        for document in corpus:
            self.doc_len.append(len(document))
            frequencies = Counter(document)
            self.doc_freqs.append(frequencies)
            for word, freq in frequencies.items():
                nd[word] = nd.get(word, 0) + 1
                
        for word, freq in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query: list[str], k1=1.5, b=0.75):
        scores = [0] * self.corpus_size
        for q in query:
            if q not in self.idf:
                continue
            idf = self.idf[q]
            for i, doc_freq in enumerate(self.doc_freqs):
                freq = doc_freq.get(q, 0)
                if freq == 0:
                    continue
                score = idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * self.doc_len[i] / self.avgdl))
                scores[i] += score
        return scores


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: EmbeddingGenerator = None):

        self.vector_store = vector_store
        if embedder is None:
            from src.embedding_generator import EmbeddingGenerator
            embedder = EmbeddingGenerator()
        self.embedder = embedder
        self._build_bm25()

    def retrieve_top_k(self, query: str, top_k: int = 5, intent: str = "general") -> list[dict]:
        return self.retrieve(query, top_k=top_k, intent=intent)
        
    def _build_bm25(self):
        chunks = self.vector_store.get_all_chunks()
        if not chunks:
            self.bm25 = None
            self.chunks = []
            return
            
        self.chunks = chunks
        tokenized_corpus = [self._tokenize(c.get("text", "")) for c in chunks]
        self.bm25 = SimpleBM25(tokenized_corpus)
        
    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())

    @staticmethod
    def _normalize_section_query(query: str) -> tuple[str, str]:
        """Return (ipc_section, bns_section) if present in query, else ('','')."""
        if not query:
            return "", ""
        q = str(query)

        ipc = ""
        bns = ""
        # IPC patterns
        m_ipc = re.search(r"\b(?:ipc)\s*(?:section|sec)?\s*(\d+[A-Z]*)\b", q, flags=re.IGNORECASE)
        if m_ipc:
            ipc = VectorStore.normalize_section(m_ipc.group(1))
        # BNS patterns
        m_bns = re.search(r"\b(?:bns)\s*(?:section|sec)?\s*(\d+[A-Z]*)\b", q, flags=re.IGNORECASE)
        if m_bns:
            bns = VectorStore.normalize_section(m_bns.group(1))

        # Generic "section 403" or "sec 403" (could be IPC)
        m_generic = re.search(r"\b(?:section|sec)\s*(\d+[A-Z]*)\b", q, flags=re.IGNORECASE)
        if m_generic and not ipc:
            ipc = VectorStore.normalize_section(m_generic.group(1))

        # Plain "403" is ambiguous; only accept if query contains 'ipc'/'section' keywords.
        return ipc, bns

    def retrieve(self, query: str, top_k: int = 20, intent: str = "general") -> list[dict]:

        """Hybrid retrieval with strong exact-match boosting.

        Exact retrieval targets:
        - IPC section
        - BNS section
        - crime name
        Then merges:
        - FAISS semantic search
        - BM25 keyword search
        """
        try:
            if self.vector_store.index is None or self.vector_store.index.ntotal == 0:
                return []

            combined_dict: dict[Any, dict] = {}

            # ---- Exact searches (hard boost) ----
            ipc_norm, bns_norm = self._normalize_section_query(query)

            if ipc_norm:
                for r in self.vector_store.exact_search_ipc(ipc_norm):
                    cid = r["chunk_id"]
                    combined_dict.setdefault(cid, r.copy())
                    combined_dict[cid]["final_score"] = min(combined_dict[cid].get("final_score", 0) + r.get("final_score", 100.0), 100.0)

            if bns_norm:
                for r in self.vector_store.exact_search_bns(bns_norm):
                    cid = r["chunk_id"]
                    combined_dict.setdefault(cid, r.copy())
                    combined_dict[cid]["final_score"] = min(combined_dict[cid].get("final_score", 0) + r.get("final_score", 100.0), 100.0)

            # If query contains only a generic 'section 403' treat as IPC.
            if not ipc_norm and not bns_norm:
                # Try a last resort extraction: "403" only when query mentions 'section/sec/ipc'
                if re.search(r"\b(ipc|section|sec)\b", query, flags=re.IGNORECASE):
                    m = re.search(r"(\d+[A-Z]*)", query)
                    if m:
                        ipc_norm = self.vector_store.normalize_section(m.group(1))
                        for r in self.vector_store.exact_search_ipc(ipc_norm):
                            cid = r["chunk_id"]
                            combined_dict.setdefault(cid, r.copy())
                            combined_dict[cid]["final_score"] = min(combined_dict[cid].get("final_score", 0) + r.get("final_score", 100.0), 100.0)


            # Crime name exact-ish match
            crime_results = self.vector_store.exact_search_crime_name(query)
            for r in crime_results:
                cid = r["chunk_id"]
                combined_dict.setdefault(cid, r.copy())
                combined_dict[cid]["final_score"] = min(combined_dict[cid].get("final_score", 0) + r.get("final_score", 60.0), 100.0)


            # ---- FAISS semantic search ----
            embeddings = self.embedder.generate_embeddings([query])
            query_embedding = embeddings[0]
            faiss_results = self.vector_store.search(query_embedding, top_k=20)
            for r in faiss_results:
                cid = r["chunk_id"]
                combined_dict.setdefault(cid, r.copy())
                # Normalize and accumulate safely
                score_boost = r.get("similarity", 0) * 0.35
                if intent in ["definition", "scenario"]:
                    score_boost *= 1.2 # boost semantic for natural queries
                    
                combined_dict[cid]["final_score"] = min(
                    combined_dict[cid].get("final_score", 0) + score_boost,
                    100.0
                )

            # ---- BM25 keyword search ----
            if self.bm25:
                tokenized_query = self._tokenize(query)
                scores = self.bm25.get_scores(tokenized_query)
                top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:20]
                for i in top_indices:
                    if scores[i] <= 0:
                        continue
                    res = self.chunks[i]
                    cid = res["chunk_id"]
                    combined_dict.setdefault(cid, res.copy())
                    norm_bm25 = min(scores[i] / 15.0, 1.0)
                    score_boost = norm_bm25 * 0.25
                    
                    if intent in ["punishment", "procedure", "section_lookup"]:
                        score_boost *= 1.2 # boost exact keywords
                        
                    combined_dict[cid]["final_score"] = min(
                        combined_dict[cid].get("final_score", 0) + score_boost,
                        100.0
                    )

            final_list = list(combined_dict.values())
            final_list.sort(key=lambda x: x.get("final_score", 0), reverse=True)

            for r in final_list:
                r["similarity"] = min(float(r.get("final_score", 0)), 1.0)

            return final_list[:top_k]
        except Exception as e:
            import logging
            logger = logging.getLogger("rag_engine")
            logger.exception(e)
            return []

