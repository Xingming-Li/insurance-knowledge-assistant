"""Retrieval interface for the Insurance Knowledge Assistant

This version does dense similarity retrieval with a configurable top-k, plus a
minimal species-safety filter: when the question clearly names a species, the
OTHER species' TERMS document is excluded so the near-duplicate dog/cat terms
cannot contaminate results. General documents always remain retrievable.

The class is structured so a reranker, hybrid (keyword + dense) search, or query
rewriting can be added later without changing the calling code.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional

from config import Settings, get_settings

DOG_TERMS_DOC = "NP-DOG-TERMS-2026"
CAT_TERMS_DOC = "NP-CAT-TERMS-2026"

# Deterministic (no LLM) lexical species detection. Word-boundary anchored so
# "hundra" (hundred) does not match "hund", and covers common Swedish forms
# incl. compounds like "hundförsäkring"/"kattförsäkring".
_DOG_RE = re.compile(r"\bhund(?:en|ar|ars|arna|arnas|s|försäkring\w*)?\b", re.IGNORECASE)
_CAT_RE = re.compile(r"\bkatt(?:en|er|ers|erna|ernas|s|försäkring\w*)?\b", re.IGNORECASE)


def detect_species(text: str) -> Optional[str]:
    """Return 'dog', 'cat', or None (None if ambiguous or undetermined)."""
    has_dog = bool(_DOG_RE.search(text or ""))
    has_cat = bool(_CAT_RE.search(text or ""))
    if has_dog and not has_cat:
        return "dog"
    if has_cat and not has_dog:
        return "cat"
    return None  # both or neither -> no species filter


def species_filter(question: str) -> Optional[dict]:
    """Chroma ``where`` excluding only the OTHER species' TERMS document.

    Excludes the opposite terms doc; keeps the matching terms doc AND all
    general docs (VET/CLAIMS/EXCL/CS). Returns None when species is undetermined.
    """
    species = detect_species(question)
    if species == "dog":
        return {"document_id": {"$nin": [CAT_TERMS_DOC]}}
    if species == "cat":
        return {"document_id": {"$nin": [DOG_TERMS_DOC]}}
    return None


# --------------------------------------------------------------------------
# Deterministic supplementary retrieval for deductible calculations
# --------------------------------------------------------------------------

# The deductible-mechanics section is identically titled in both terms docs.
MECHANICS_SECTION = "5. Självrisker"

# Cues that a question asks for an EXACT amount calculation, paired with a
# reimbursement/deductible term. Purely lexical; no amounts or formulas here.
_CALC_CUES = ("hur mycket", "vad blir", "hur stor", "beräkn", "räkna ut")
_CALC_TERMS = ("ersätt", "självrisk", "betalar", "egen kostnad", "utbetal")


def is_calculation_question(text: str) -> bool:
    """True when the question requests an exact reimbursement/deductible/cost sum."""
    t = (text or "").lower()
    return any(c in t for c in _CALC_CUES) and any(term in t for term in _CALC_TERMS)


def _fetch_section(vector_store: Any, document_id: str, section: str):
    """Deterministically fetch a chunk by metadata (document_id + section)."""
    res = vector_store.get(
        where={"$and": [{"document_id": document_id}, {"section": section}]}
    )
    documents = (res or {}).get("documents") or []
    metadatas = (res or {}).get("metadatas") or []
    if not documents:
        return None
    from langchain_core.documents import Document

    return Document(page_content=documents[0], metadata=metadatas[0] or {})


def supplement_deductible_mechanics(
    question: str,
    docs: List[Any],
    vector_store: Any,
    max_chunks: int = 8,
) -> List[Any]:
    """Ensure the applicable §5 Självrisker chunk is in the evidence.

    Only acts when the question is a calculation request AND species is dog/cat.
    If the matching §5 is missing, fetch it by metadata and append it.

    Normal retrieval may return up to ``max_chunks`` chunks. For calculation
    questions that require supplementation, the final evidence set may contain
    up to ``max_chunks + 1`` chunks so that an existing relevant result is not
    evicted.

    Deduplicated by (document_id, section). Knows only that deductible
    calculations need the applicable deductible-mechanics section — no amounts,
    rates or formulas.
    """
    if not is_calculation_question(question):
        return docs

    species = detect_species(question)
    if species not in ("dog", "cat"):
        return docs

    target_doc = DOG_TERMS_DOC if species == "dog" else CAT_TERMS_DOC

    # Already present: leave the original retrieval unchanged.
    for d in docs:
        m = d.metadata or {}
        if (
            m.get("document_id") == target_doc
            and m.get("section") == MECHANICS_SECTION
        ):
            return docs

    chunk = _fetch_section(vector_store, target_doc, MECHANICS_SECTION)
    if chunk is None:
        return docs  # prompt rule remains the second line of defense

    result = list(docs)
    result.append(chunk)

    # Deduplicate while preserving retrieval order.
    seen = set()
    out = []

    for d in result:
        m = d.metadata or {}
        key = (m.get("document_id"), m.get("section"))

        if key in seen:
            continue

        seen.add(key)
        out.append(d)

    # Normal retrieval: max_chunks.
    # Calculation supplementation: allow one additional evidence chunk.
    return out[: max_chunks + 1]


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
        """Return the top-k relevant chunks (as Documents) for ``question``.

        Applies the species-safety filter when the question names a species.
        """
        k = k or self.settings.retrieval_k
        flt = species_filter(question)
        return self._get_vector_store().similarity_search(question, k=k, filter=flt)

    def retrieve_for_answer(self, question: str, k: Optional[int] = None) -> List[Any]:
        """Normal retrieval + deterministic deductible-mechanics supplementation.

        This is the pipeline entry point used by the app: it does not change the
        dense algorithm, only guarantees the applicable §5 chunk is present for
        calculation questions.
        """
        k = k or self.settings.retrieval_k
        docs = self.retrieve(question, k=k)
        return supplement_deductible_mechanics(
            question, docs, self._get_vector_store(), max_chunks=k
        )

    # --- Extension points for later versions -----------------------------
    # def _rerank(self, question, docs): ...
    # def _hybrid_search(self, question, k): ...
    # def _rewrite_query(self, question): ...
