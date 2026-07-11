"""
Topic Modeling Module
====================
Extract hidden legal topics using BERTopic.

If BERTopic is unavailable, falls back to an NMF-based topic model.

BERTopic combines:
- Sentence Transformers for embeddings
- UMAP for dimensionality reduction
- HDBSCAN for density-based clustering
- C-TF-IDF for keyword extraction

Fallback (no BERTopic):
- TF-IDF vectors from raw documents (embeddings are ignored)
- NMF to discover topics
- Topic keywords from top NMF components

Both paths expose a consistent interface:
- TopicModeler.fit()
- TopicModeler.topics
- TopicModeler.probabilities (if available)
- TopicModeler.get_topic_info()
- TopicModeler.to_dict()
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple, Optional

import numpy as np


class TopicModeler:
    """Topic modeling wrapper (BERTopic when available, otherwise NMF)."""

    def __init__(self, embeddings: np.ndarray, documents: Optional[List[str]] = None):
        self.embeddings = embeddings
        self.documents = documents

        self.n_docs = int(embeddings.shape[0]) if embeddings is not None else 0

        self.model = None
        self.topics: Optional[np.ndarray] = None
        self.probabilities: Optional[np.ndarray] = None

        print(f"📚 Initialized topic modeler")
        print(f"   Documents: {self.n_docs}")

    def _fit_bertopic(self, language: str = "english") -> None:
        try:
            from bertopic import BERTopic  # type: ignore
        except Exception as e:
            raise ImportError("BERTopic is not available") from e

        print("\n🔍 Fitting BERTopic model...")

        self.model = BERTopic(
            language=language,
            calculate_probabilities=True,
            verbose=True,
        )

        self.topics, self.probabilities = self.model.fit_transform(
            documents=None,
            embeddings=self.embeddings,
        )

        n_topics = len(set(self.topics)) - (1 if -1 in self.topics else 0)
        print(f"\n✓ Discovered {n_topics} topics")

    def _fit_nmf(self, language: str = "english") -> None:
        """Simple NMF topic modeling fallback.

        Requires `documents` to be provided.
        """

        if not self.documents:
            raise ValueError("NMF fallback requires `documents` (raw texts) to be provided.")

        from sklearn.decomposition import NMF
        from sklearn.feature_extraction.text import TfidfVectorizer

        print("\n🔍 BERTopic unavailable -> using NMF fallback...")

        # TF-IDF from raw docs
        vectorizer = TfidfVectorizer(
            stop_words=language,
            max_df=0.95,
            min_df=1,
            ngram_range=(1, 2),
        )
        X = vectorizer.fit_transform(self.documents)

        # Heuristic topic count: sqrt(n/2), bounded.
        # NMF requires n_components <= min(n_samples, n_features).
        # Here, n_samples = number of documents; n_features = TF-IDF vocab size.
        X = X  # tf-idf matrix
        n_samples = X.shape[0]
        n_features = X.shape[1]
        k = int(max(2, min(10, round(np.sqrt(max(2, self.n_docs) / 2)))))
        k = max(2, min(k, n_samples, n_features))

        # Ensure k is valid for sklearn init='nndsvda'
        if k > min(n_samples, n_features):
            k = min(n_samples, n_features)
        if k < 2:
            k = 2

        model = NMF(
            n_components=k,

            init="nndsvda",
            random_state=42,
            max_iter=500,
        )

        W = model.fit_transform(X)  # document-topic weights
        H = model.components_  # topic-term weights

        # Assign topic as max component
        self.topics = np.argmax(W, axis=1)

        # Create pseudo-probabilities by normalizing W per document
        denom = np.sum(W, axis=1, keepdims=True) + 1e-12
        probs = W / denom
        self.probabilities = probs

        # Wrap model so get_topic_info can reuse
        self.model = {
            "vectorizer": vectorizer,
            "nmf": model,
            "topic_terms": vectorizer.get_feature_names_out(),
            "H": H,
        }

        print(f"✓ NMF discovered {k} topics")

    def fit(self, language: str = "english") -> None:
        # Prefer BERTopic if importable, else fallback
        try:
            self._fit_bertopic(language=language)
        except Exception:
            self._fit_nmf(language=language)

    def get_topic_info(self) -> Dict:
        if self.topics is None:
            raise ValueError("Must fit model first")

        topic_info: Dict[int, Dict] = {}

        # BERTopic path
        if hasattr(self.model, "get_topic"):
            topics_list = sorted(set(self.topics))
            if -1 in topics_list:
                topics_list.remove(-1)

            for topic_id in topics_list:
                words_weights = self.model.get_topic(topic_id)
                words = [word for word, _ in words_weights]
                weights = [float(weight) for _, weight in words_weights]

                topic_label = self.model.generate_topic_labels(
                    nr_words=3,
                    topic_list=[topic_id],
                )[topic_id]

                topic_docs = np.where(self.topics == topic_id)[0]

                topic_info[int(topic_id)] = {
                    "label": topic_label,
                    "keywords": words[:10],
                    "weights": weights[:10],
                    "document_count": int(len(topic_docs)),
                    "document_percentage": float(len(topic_docs) / self.n_docs * 100),
                }

            return topic_info

        # NMF fallback path
        vec = self.model["vectorizer"]
        topic_terms = self.model["topic_terms"]
        H = self.model["H"]

        for topic_id in sorted(set(self.topics)):
            # top terms for each topic
            comp = H[topic_id]
            top_idx = np.argsort(comp)[::-1][:10]
            words = [str(topic_terms[i]) for i in top_idx]
            weights = [float(comp[i]) for i in top_idx]

            topic_docs = np.where(self.topics == topic_id)[0]
            topic_info[int(topic_id)] = {
                "label": f"Topic {topic_id}",
                "keywords": words,
                "weights": weights,
                "document_count": int(len(topic_docs)),
                "document_percentage": float(len(topic_docs) / self.n_docs * 100),
            }

        return topic_info

    def get_top_topics(self, n: int = 5) -> Dict:
        topic_info = self.get_topic_info()
        sorted_topics = sorted(
            topic_info.items(),
            key=lambda x: x[1]["document_count"],
            reverse=True,
        )
        return {k: v for k, v in sorted_topics[:n]}

    def get_document_topics(self, doc_index: int) -> Dict:
        if self.topics is None:
            raise ValueError("Must fit model first")

        primary_topic = int(self.topics[doc_index])

        top_topics = []
        if self.probabilities is not None:
            doc_probs = self.probabilities[doc_index]
            top_indices = np.argsort(doc_probs)[-3:][::-1]
            for idx in top_indices:
                top_topics.append({"topic_id": int(idx), "probability": float(doc_probs[idx])})
        else:
            top_topics = [{"topic_id": primary_topic, "probability": 1.0}]

        return {
            "document_index": doc_index,
            "primary_topic": primary_topic,
            "top_topics": top_topics,
        }

    def get_topic_distribution(self) -> Dict:
        if self.topics is None:
            raise ValueError("Must fit model first")

        unique_topics, counts = np.unique(self.topics, return_counts=True)
        distribution: Dict[int, Dict] = {}
        for topic_id, count in zip(unique_topics, counts):
            if topic_id == -1:
                continue
            distribution[int(topic_id)] = {
                "count": int(count),
                "percentage": float(count / self.n_docs * 100),
            }
        return distribution

    def to_dict(self) -> Dict:
        if self.topics is None:
            raise ValueError("Must fit model first")

        return {
            "statistics": self.get_statistics(),
            "topic_assignments": self.topics.tolist(),
            "topic_probabilities": self.probabilities.tolist() if self.probabilities is not None else None,
        }

    def get_statistics(self) -> Dict:
        if self.topics is None:
            raise ValueError("Must fit model first")

        n_topics = len(set(self.topics)) - (1 if -1 in self.topics else 0)
        return {
            "n_documents": int(self.n_docs),
            "n_topics": int(n_topics),
            "n_outliers": int(np.sum(self.topics == -1)),
            "topic_distribution": self.get_topic_distribution(),
            "topics": self.get_topic_info(),
        }
    def reduce_outliers(self):
        """
        BERTopic compatibility method.

        BERTopic supports outlier reduction, but the NMF fallback
        does not produce outlier topics. This method exists to
        prevent pipeline crashes when BERTopic is unavailable.
        """

        print("ℹ️ Outlier reduction skipped (NMF fallback mode)")
        return self
    # Keep BERTopic save/load for BERTopic path only.
    def save_model(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Must fit model first")
        if hasattr(self.model, "save"):
            self.model.save(path)
            print(f"💾 Model saved to {path}")
        else:
            # NMF fallback model dict isn't saved here; pipeline can persist `to_dict()`.
            raise NotImplementedError("NMF fallback model persistence not implemented")

    def load_model(self, path: str) -> None:
        from bertopic import BERTopic  # type: ignore

        self.model = BERTopic.load(path)
        print(f"📂 Model loaded from {path}")

