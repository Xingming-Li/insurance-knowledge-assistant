"""Centralised configuration for the Insurance Knowledge Assistant.

All tunables come from environment variables (with sensible defaults) so that
nothing sensitive — above all the OpenAI API key — is hard-coded. Call
``get_settings()`` to obtain an immutable snapshot of the current configuration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Load a local .env file if python-dotenv is available. Optional so that the
# module (and the unit tests) import cleanly even without the dependency.
try:  # pragma: no cover - trivial import guard
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

# Project root = parent of the ``src`` directory that holds this file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCS = PROJECT_ROOT / "data" / "insurance_docs"
DEFAULT_CHROMA = PROJECT_ROOT / "chroma_insurance"


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return int(value)


@dataclass(frozen=True)
class Settings:
    """Immutable configuration snapshot."""

    # Secrets
    openai_api_key: Optional[str]

    # Models
    embedding_model: str
    chat_model: str

    # Vector store
    chroma_path: str
    collection_name: str

    # Chunking
    chunk_size: int
    chunk_overlap: int

    # Retrieval
    retrieval_k: int

    # Data
    docs_path: str

    def require_api_key(self) -> str:
        """Return the API key or raise if it has not been configured."""
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your environment or a .env file."
            )
        return self.openai_api_key


def get_settings() -> Settings:
    """Build a :class:`Settings` snapshot from the current environment."""
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        chroma_path=os.getenv("CHROMA_PATH", str(DEFAULT_CHROMA)),
        collection_name=os.getenv("CHROMA_COLLECTION", "insurance_docs"),
        chunk_size=_int_env("CHUNK_SIZE", 1000),
        chunk_overlap=_int_env("CHUNK_OVERLAP", 120),  # ~12% overlap
        retrieval_k=_int_env("RETRIEVAL_K", 4),
        docs_path=os.getenv("DOCS_PATH", str(DEFAULT_DOCS)),
    )
