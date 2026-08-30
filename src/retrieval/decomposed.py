"""Experimental decomposed retrieval path.

This does NOT replace ``InsuranceRetriever`` (the dense baseline). It adds an
alternative strategy for compound insurance questions:

  1. decompose a compound question into a few evidence-oriented subqueries
     (retrieval queries, NOT answers), each tagged with a scope;
  2. retrieve a small number of chunks per subquery, applying SELECTIVE
     species metadata filtering to avoid dog/cat near-duplicate contamination;
  3. merge, deduplicate and bound the final evidence set.

Decomposition and vector store are dependency-injectable so tests run without
any API calls.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from config import Settings, get_settings

DOG_TERMS = "NP-DOG-TERMS-2026"
CAT_TERMS = "NP-CAT-TERMS-2026"
TERMS_DOCS = [DOG_TERMS, CAT_TERMS]

# A decomposer is any callable (question) -> {"species", "subqueries": [...]}.
Decomposer = Callable[[str], Dict[str, Any]]


# --------------------------------------------------------------------------
# Selective species metadata filtering
# --------------------------------------------------------------------------

def build_filter(species: Optional[str], scope: str) -> Optional[Dict[str, Any]]:
    """Chroma ``where`` filter for a subquery.

    * scope="species_terms": restrict species-specific TERMS facts to the one
      matching product doc (dog->DOG terms, cat->CAT terms). This is what stops
      the near-duplicate dog/cat terms documents from contaminating each other.
    * scope="general": search the shared documents (veterinary, claims,
      exclusions, customer-service) — i.e. everything EXCEPT the two terms docs.

    Returns None (no filter) only when a species_terms subquery has no species.
    """
    if scope == "species_terms":
        if species == "dog":
            return {"document_id": DOG_TERMS}
        if species == "cat":
            return {"document_id": CAT_TERMS}
        return None  # unknown species: cannot restrict, fall back to no filter
    # general
    return {"document_id": {"$nin": TERMS_DOCS}}


def doc_passes_filter(document_id: str, flt: Optional[Dict[str, Any]]) -> bool:
    """Small filter matcher (used by tests and by fakes without Chroma)."""
    if not flt:
        return True
    cond = flt.get("document_id")
    if isinstance(cond, dict):
        if "$nin" in cond:
            return document_id not in cond["$nin"]
        if "$in" in cond:
            return document_id in cond["$in"]
        if "$eq" in cond:
            return document_id == cond["$eq"]
        return True
    return document_id == cond


# --------------------------------------------------------------------------
# LLM decomposer (default, injectable)
# --------------------------------------------------------------------------

_DECOMPOSER_SYSTEM = (
    "You turn a compound Swedish pet-insurance question into a SMALL set of "
    "retrieval subqueries (search queries, NOT answers). Identify the species "
    "('dog' for hund, 'cat' for katt, else null). For each distinct evidence "
    "need, produce one short Swedish subquery of keywords. Set scope to "
    "'species_terms' when the need is a species-specific policy fact from the "
    "product terms (reimbursement limits/ersättningstak, deductibles/självrisk, "
    "coverage tiers/nivåer, waiting periods/karens), and 'general' otherwise "
    "(veterinary care, surgery, pre-authorisation, claims, exclusions, customer "
    "service). Do NOT over-split: a simple single-intent question must yield "
    "exactly ONE subquery. Never produce more than four subqueries."
)


class QueryDecomposer:
    """Default LLM-backed decomposer. Reuses the chat model, temperature 0."""

    def __init__(self, settings: Optional[Settings] = None, model: Optional[str] = None):
        self.settings = settings or get_settings()
        self.model = model or self.settings.chat_model
        self._chain = None

    def _get_chain(self):
        if self._chain is None:
            from typing import List as L
            from typing import Optional as O

            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI
            from pydantic import BaseModel, Field

            class SubQuery(BaseModel):
                query: str = Field(description="Short Swedish retrieval query (keywords)")
                scope: str = Field(description="'species_terms' or 'general'")

            class Decomposition(BaseModel):
                species: O[str] = Field(None, description="'dog', 'cat' or null")
                subqueries: L[SubQuery] = Field(description="1-4 retrieval subqueries")

            llm = ChatOpenAI(
                model=self.model,
                temperature=self.settings.chat_temperature,
                api_key=self.settings.require_api_key(),
            )
            prompt = ChatPromptTemplate.from_messages(
                [("system", _DECOMPOSER_SYSTEM), ("human", "{question}")]
            )
            self._chain = prompt | llm.with_structured_output(Decomposition)
        return self._chain

    def __call__(self, question: str) -> Dict[str, Any]:
        d = self._get_chain().invoke({"question": question})
        species = d.species if d.species in ("dog", "cat") else None
        subqueries = [{"query": s.query, "scope": s.scope} for s in d.subqueries]
        return {"species": species, "subqueries": subqueries}


# --------------------------------------------------------------------------
# Decomposed retriever
# --------------------------------------------------------------------------

class DecomposedRetriever:
    """Compound-question retriever: decompose -> filtered subquery search -> merge."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        vector_store: Any = None,
        decomposer: Optional[Decomposer] = None,
        k_sub: int = 3,
        max_chunks: int = 8,
        max_subqueries: int = 4,
    ):
        self.settings = settings or get_settings()
        self._vector_store = vector_store
        self._decomposer = decomposer
        self.k_sub = k_sub
        self.max_chunks = max_chunks
        self.max_subqueries = max_subqueries

    # -- lazily built collaborators -------------------------------------
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

    def _get_decomposer(self) -> Decomposer:
        if self._decomposer is None:
            self._decomposer = QueryDecomposer(settings=self.settings)
        return self._decomposer

    # -- core -----------------------------------------------------------
    def retrieve_detailed(self, question: str) -> Dict[str, Any]:
        """Return decomposition, per-subquery hits and the merged evidence set."""
        decomposition = self._get_decomposer()(question)
        species = decomposition.get("species")
        subqueries = decomposition.get("subqueries") or []
        if not subqueries:  # robustness: fall back to the whole question
            subqueries = [{"query": question, "scope": "general"}]
        subqueries = subqueries[: self.max_subqueries]

        vs = self._get_vector_store()
        hits = []  # dicts with doc + provenance
        for idx, sq in enumerate(subqueries):
            flt = build_filter(species, sq.get("scope", "general"))
            results = vs.similarity_search_with_relevance_scores(
                sq["query"], k=self.k_sub, filter=flt
            )
            for doc, score in results:
                m = doc.metadata or {}
                hits.append(
                    {
                        "doc": doc,
                        "score": score,
                        "document_id": m.get("document_id"),
                        "section": m.get("section"),
                        "start_index": m.get("start_index"),
                        "subquery_idx": idx,
                        "subquery": sq["query"],
                        "scope": sq.get("scope", "general"),
                        "filter": flt,
                    }
                )

        merged = self._merge(hits)
        return {
            "decomposition": decomposition,
            "subqueries": subqueries,
            "hits": hits,
            "final": merged,
        }

    def retrieve(self, question: str, k: Optional[int] = None) -> List[Any]:
        """Return the merged evidence documents (interface-compatible)."""
        detailed = self.retrieve_detailed(question)
        docs = [h["doc"] for h in detailed["final"]]
        return docs[:k] if k else docs

    def _merge(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate by (document_id, section), keep best score, then bound.

        Deduplicating on (document_id, section) both removes identical chunks
        surfaced by multiple subqueries and avoids repeated document/section
        duplicates in the final set.
        """
        best: Dict[tuple, Dict[str, Any]] = {}
        for h in hits:
            key = (h["document_id"], h["section"])
            if key not in best or h["score"] > best[key]["score"]:
                best[key] = h
        ordered = sorted(best.values(), key=lambda h: h["score"], reverse=True)
        return ordered[: self.max_chunks]
