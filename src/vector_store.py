"""Smart Legal Assistance - Phase 2 Module 1

FAISS VectorStore
------------------
Builds and loads a local FAISS index (IndexFlatL2) over embeddings.

This implementation supports BOTH:
- New record-based schema (ipc_section/bns_section fields)
- Old schema (section_number field)

It provides robust exact search and section normalization.
"""

from __future__ import annotations

import os
import pickle
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("faiss-cpu is required for VectorStore. Install via requirements.txt") from e

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS vector store for semantic retrieval."""

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        embedding_dim: Optional[int] = None,
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_dim = embedding_dim

        self.index: Optional[Any] = None
        self.metadata: Optional[List[Dict[str, Any]]] = None

    def build_index(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        embeddings = np.asarray(embeddings)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape={embeddings.shape}")

        n, d = embeddings.shape
        if self.embedding_dim is None:
            self.embedding_dim = d
        if d != self.embedding_dim:
            raise ValueError(f"Embedding dim mismatch. Expected {self.embedding_dim}, got {d}")

        if len(metadata) != n:
            raise ValueError(
                f"metadata length mismatch: embeddings has {n} vectors but metadata has {len(metadata)}"
            )

        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = metadata

        logger.info("Building FAISS IndexFlatL2 with %d vectors (dim=%d)", n, d)

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        if self.index is None or self.metadata is None:
            self.build_index(embeddings, metadata)
            return

        embeddings = np.asarray(embeddings)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D")

        n, d = embeddings.shape
        if self.embedding_dim is not None and d != self.embedding_dim:
            raise ValueError(f"Dim mismatch. Expected {self.embedding_dim}, got {d}")

        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadata)
        logger.info("Added %d new vectors to FAISS index", n)

    def save_index(self) -> None:
        if self.index is None or self.metadata is None:
            raise ValueError("Index/metadata not built. Call build_index first.")

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        logger.info("Saved FAISS index to %s", self.index_path)
        logger.info("Saved metadata to %s", self.metadata_path)

    def load_index(self) -> None:
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        if self.metadata is None:
            raise ValueError("Loaded metadata is None")

        # Enforce metadata schema (client-required) at load time.
        self.metadata = [self._ensure_metadata_schema(m) for m in self.metadata]

        if self.embedding_dim is None:
            self.embedding_dim = int(self.index.d)

        logger.info("Loaded FAISS index from %s", self.index_path)
        logger.info("Loaded metadata (%d items)", len(self.metadata))

        print("\n" + "=" * 50)
        print("STEP 3 : VERIFY FAISS INDEX")
        print("=" * 50)
        print(f"Number of vectors: {self.index.ntotal}")
        print(f"Verified files: {self.index_path}, {self.metadata_path}")
        print("=" * 50 + "\n")

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        return float(1.0 / (1.0 + float(distance)))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or self.metadata is None:
            raise ValueError("Index not loaded/built. Call load_index() or build_index().")

        q = np.asarray(query_embedding)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[0] != 1:
            raise ValueError("query_embedding must have shape (dim,) or (1, dim)")
        if q.shape[1] != self.index.d:
            raise ValueError(f"Query dim mismatch: index dim={self.index.d}, query dim={q.shape[1]}")

        distances, indices = self.index.search(q.astype(np.float32), int(top_k))
        distances = distances[0]
        indices = indices[0]

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances, indices):
            if idx < 0:
                continue
            m = self.metadata[int(idx)]
            results.append({
                **m,
                "distance": float(dist),
                "similarity": self._distance_to_similarity(dist),
            })

        return results

    @staticmethod
    def normalize_section(raw: Any) -> str:
        """Normalize section identifiers to a canonical digits/letter token.

        Examples:
          "403", "section 403", "IPC 403", "Sec 403" -> "403"
          "314", "BNS Section 314" -> "314"
        """
        if raw is None:
            return ""
        s = str(raw).strip()
        if not s:
            return ""
        m = re.search(r"(\d+[A-Z]*)", s, flags=re.IGNORECASE)
        return m.group(1).upper() if m else ""

    @staticmethod
    def _ensure_metadata_schema(m: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure each metadata entry matches the required schema for legal RAG."""
        ipc_section = m.get("ipc_section", "")
        bns_section = m.get("bns_section", "")
        if not ipc_section and "section_number" in m:
            ipc_section = m.get("section_number", "")

        ipc_section = VectorStore.normalize_section(ipc_section)
        bns_section = VectorStore.normalize_section(bns_section)

        crime_name = str(m.get("crime_name", "") or "").strip()
        description = str(m.get("description", "") or "").strip()
        punishment = str(m.get("punishment", "") or "").strip()

        # Reconstruct text if missing or empty.
        text = m.get("text", "") or ""
        if not str(text).strip():
            text = (
                f"IPC Section: {ipc_section}\n"
                f"Crime: {crime_name}\n"
                f"Description: {description}\n"
                f"Punishment: {punishment}\n"
                f"BNS Section: {bns_section}"
            ).strip()

        chunk_id = m.get("chunk_id", 0)
        try:
            chunk_id = int(chunk_id)
        except (ValueError, TypeError):
            # Keep as string if it's a UUID or other string format
            chunk_id = str(chunk_id)

        return {
            "chunk_id": chunk_id,
            "source_document": str(m.get("source_document", "") or "").strip(),
            "ipc_section": ipc_section,
            "bns_section": bns_section,
            "crime_name": crime_name,
            "description": description,
            "punishment": punishment,
            "keywords": m.get("keywords", []),
            "page_number": m.get("page_number", 1),
            "text": str(text).strip(),
            "aliases": m.get("aliases", []),
            "related_sections": m.get("related_sections", "N/A"),
            "complete_text": m.get("complete_text", ""),
        }

    def _normalize_doc_section(self, m: Dict[str, Any], field: str) -> str:
        # field can be ipc_section or bns_section in new schema
        if field in m:
            return self.normalize_section(m.get(field, ""))

        # fallback for old schema
        if field == "ipc_section" and "section_number" in m:
            return self.normalize_section(m.get("section_number", ""))

        return ""

    def _exact_search_field(self, field: str, query_value: str) -> List[Dict[str, Any]]:
        if self.metadata is None:
            return []
        qn = self.normalize_section(query_value)
        if not qn:
            return []

        results: List[Dict[str, Any]] = []
        for m in self.metadata:
            mv = self._normalize_doc_section(m, field)
            if mv and mv == qn:
                results.append({
                    **m,
                    "distance": 0.0,
                    "similarity": 1.0,
                    "final_score": 100.0,
                })
        return results

    def exact_search_ipc(self, ipc_section: str) -> List[Dict[str, Any]]:
        return self._exact_search_field("ipc_section", ipc_section)

    def exact_search_bns(self, bns_section: str) -> List[Dict[str, Any]]:
        return self._exact_search_field("bns_section", bns_section)

    def exact_search_crime_name(self, crime_name_query: str) -> List[Dict[str, Any]]:
        if self.metadata is None:
            return []
        q = str(crime_name_query or "").strip().lower()
        if not q:
            return []

        results: List[Dict[str, Any]] = []
        for m in self.metadata:
            cn = str(m.get("crime_name", "")).strip().lower()
            if not cn:
                continue
            if cn == q or q in cn or cn in q:
                results.append({
                    **m,
                    "distance": 0.0,
                    "similarity": 1.0,
                    "final_score": 60.0,
                })
        return results

    # Backwards compatibility with old callers
    def exact_search(self, section_number: str) -> List[Dict[str, Any]]:
        return self.exact_search_ipc(section_number)

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        return self.metadata or []

    def validate_index(self):
        """Validates the health and content of the FAISS index and metadata."""
        if self.index is None:
            print("Index is not loaded.")
            return
            
        print("\n=== FAISS Index Validation ===")
        print(f"Total Vectors: {self.index.ntotal}")
        print(f"Total Metadata Records: {len(self.metadata)}")
        print(f"Dimension: {self.index.d}")
        print(f"Match (Vectors == Metadata): {self.index.ntotal == len(self.metadata)}")
        
        ipc_count = sum(1 for m in self.metadata if m.get("ipc_section"))
        bns_count = sum(1 for m in self.metadata if m.get("bns_section"))
        crime_count = sum(1 for m in self.metadata if m.get("crime_name"))
        
        print(f"Records with IPC Section: {ipc_count} ({(ipc_count/max(1, len(self.metadata)))*100:.1f}%)")
        print(f"Records with BNS Section: {bns_count} ({(bns_count/max(1, len(self.metadata)))*100:.1f}%)")
        print(f"Records with Crime Name: {crime_count} ({(crime_count/max(1, len(self.metadata)))*100:.1f}%)")
        print("==============================\n")
