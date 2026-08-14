"""Retrieval interface for the Insurance Knowledge Assistant

This version does dense similarity retrieval with a configurable top-k.
The class is structured so a reranker, hybrid (keyword + dense) search,
or query rewriting can be added later without changing the calling code.
"""
from __future__ import annotations

from typing import Any, List, Optional

from config import Settings, get_settings


class InsuranceRetriever:
    """Returns the most relevant document chunks for a (Swedish) question."""

    def __init__(self, settings: Optional[Settings] = None, vector_store: Any = None):
        self.settings = settings or get_settings()
        # ``vector_store`` can be injected (e.g. in tests). Otherwise it is
        # created lazily on first use from Chroma + OpenAI embeddings.
        self._vector_store = vector_store

    def _get_vector_store(self) -> Any:
        if self._vector_store is None:
            from langchain_chroma import Chroma
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(
                model=self.settings.embedding_model,
                api_key=self.settings.require_api_key(),
            )
            self._vector_store = Chroma(
                collection_name=self.settings.collection_name,
                embedding_function=embeddings,
                persist_directory=self.settings.chroma_path,
            )
        return self._vector_store

    def retrieve(self, question: str, k: Optional[int] = None) -> List[Any]:
        """Return the top-k relevant chunks (as Documents) for ``question``."""
        k = k or self.settings.retrieval_k
        return self._get_vector_store().similarity_search(question, k=k)

    # --- Extension points for later versions -----------------------------
    # def _rerank(self, question, docs): ...
    # def _hybrid_search(self, question, k): ...
    # def _rewrite_query(self, question): ...
