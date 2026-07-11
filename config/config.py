"""Smart Legal Assistance - Phase 2 configuration.

Backend-only modules use these constants/paths.

This file is intentionally dependency-free and safe to import.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _repo_root() -> str:
    """Return repository root (where `processed_data/` lives)."""
    # config.py -> Smart_Legal_Assistance/config/config.py
    # repo root is two levels up from Smart_Legal_Assistance/
    this_dir = os.path.dirname(os.path.abspath(__file__))
    smart_dir = os.path.dirname(this_dir)  # Smart_Legal_Assistance
    return os.path.dirname(smart_dir)


@dataclass(frozen=True)
class Config:
    # Models
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    SUMMARIZER_MODEL: str = "t5-small"

    # Retrieval
    TOP_K: int = 5

    # Storage locations (FAISS index + metadata)
    # Note: metadata.pkl is required for grounded answers.
    MODELS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    FAISS_INDEX_PATH: str = os.path.join(MODELS_DIR, "faiss_index.bin")
    METADATA_PATH: str = os.path.join(MODELS_DIR, "metadata.pkl")

    # Phase-1 artifacts (existing)
    PROCESSED_DATA_DIR: str = os.path.join(_repo_root(), "processed_data")
    EMBEDDINGS_PATH: str = os.path.join(PROCESSED_DATA_DIR, "embeddings.npy")
    PROCESSED_CHUNKS_PATH: str = os.path.join(PROCESSED_DATA_DIR, "processed_chunks.json")

    # Reports
    REPORTS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    REPORT_PATH: str = os.path.join(REPORTS_DIR, "phase2_evaluation_report")

    # Logging
    LOGS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    LOG_PATH: str = os.path.join(LOGS_DIR, "app.log")

    # Runtime
    RANDOM_SEED: int = 42
    # Ensure deterministic sampling where applicable


# Backwards-compatible alias if code expects module-level constants.
CONFIG = Config()

# Convenience exports (module-level)
MODEL_NAME = CONFIG.EMBEDDING_MODEL
EMBEDDING_MODEL = CONFIG.EMBEDDING_MODEL
TOP_K = CONFIG.TOP_K
FAISS_INDEX_PATH = CONFIG.FAISS_INDEX_PATH
METADATA_PATH = CONFIG.METADATA_PATH
REPORT_PATH = CONFIG.REPORT_PATH
LOG_PATH = CONFIG.LOG_PATH
RANDOM_SEED = CONFIG.RANDOM_SEED
PROCESSED_CHUNKS_PATH = CONFIG.PROCESSED_CHUNKS_PATH
EMBEDDINGS_PATH = CONFIG.EMBEDDINGS_PATH

