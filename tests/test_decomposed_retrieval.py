import config
from langchain_core.documents import Document

from retrieval.decomposed import (
    CAT_TERMS,
    DOG_TERMS,
    DecomposedRetriever,
    build_filter,
    doc_passes_filter,
)


# ---- Fakes (no API) -------------------------------------------------------

class FakeVS:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def similarity_search_with_relevance_scores(self, query, k, filter=None):
        self.calls.append({"query": query, "k": k, "filter": filter})
        passing = [
            d for d in self.docs
            if doc_passes_filter((d.metadata or {}).get("document_id"), filter)
        ]
        return [(d, 1.0 - i * 0.01) for i, d in enumerate(passing[:k])]


def _doc(doc_id, section, start=0, content="text"):
    return Document(
        page_content=content,
        metadata={"document_id": doc_id, "section": section, "start_index": start},
    )


def _decomp(species, subqueries):
    return lambda _q: {"species": species, "subqueries": subqueries}


# ---- Selective filtering (pure) -------------------------------------------

def test_dog_species_filter_excludes_cat_but_general_allows_vet():
    st = build_filter("dog", "species_terms")
    assert st == {"document_id": DOG_TERMS}
    assert doc_passes_filter(DOG_TERMS, st) is True
    assert doc_passes_filter(CAT_TERMS, st) is False        # CAT terms excluded

    gen = build_filter("dog", "general")
    assert doc_passes_filter("NP-VET-2026", gen) is True     # VET still allowed
    assert doc_passes_filter(DOG_TERMS, gen) is False
    assert doc_passes_filter(CAT_TERMS, gen) is False


def test_cat_species_filter_excludes_dog_but_general_allows_shared():
    st = build_filter("cat", "species_terms")
    assert st == {"document_id": CAT_TERMS}
    assert doc_passes_filter(DOG_TERMS, st) is False         # DOG terms excluded

    gen = build_filter("cat", "general")
    assert doc_passes_filter("NP-CLAIMS-2026", gen) is True
    assert doc_passes_filter("NP-CS-2026", gen) is True
    assert doc_passes_filter(CAT_TERMS, gen) is False


def test_dog_question_never_retrieves_cat_terms():
    vs = FakeVS([
        _doc(DOG_TERMS, "5. Självrisker"),
        _doc(CAT_TERMS, "5. Självrisker"),
        _doc("NP-VET-2026", "4. Förhandsgodkännande"),
    ])
    r = DecomposedRetriever(
        settings=config.get_settings(), vector_store=vs,
        decomposer=_decomp("dog", [
            {"query": "hund premium självrisk", "scope": "species_terms"},
            {"query": "planerad operation förhandsgodkännande", "scope": "general"},
        ]),
    )
    final = r.retrieve_detailed("q")["final"]
    doc_ids = {h["document_id"] for h in final}
    assert DOG_TERMS in doc_ids
    assert "NP-VET-2026" in doc_ids
    assert CAT_TERMS not in doc_ids


# ---- Merging / dedup ------------------------------------------------------

def test_duplicate_chunks_from_subqueries_are_deduplicated():
    shared = _doc("NP-VET-2026", "3. Kirurgi")
    vs = FakeVS([shared, _doc("NP-VET-2026", "4. Förhandsgodkännande")])
    r = DecomposedRetriever(
        settings=config.get_settings(), vector_store=vs,
        decomposer=_decomp(None, [
            {"query": "kirurgi", "scope": "general"},
            {"query": "operation ingrepp", "scope": "general"},  # returns same docs
        ]),
    )
    final = r.retrieve_detailed("q")["final"]
    keys = [(h["document_id"], h["section"]) for h in final]
    assert len(keys) == len(set(keys))                       # no duplicates
    assert ("NP-VET-2026", "3. Kirurgi") in keys


# ---- No unnecessary explosion / cap --------------------------------------

def test_single_intent_question_does_not_explode():
    vs = FakeVS([_doc("NP-DOG-TERMS-2026", "6. Karenstider")])
    r = DecomposedRetriever(
        settings=config.get_settings(), vector_store=vs,
        decomposer=_decomp("dog", [{"query": "hund karens sjukdom", "scope": "species_terms"}]),
    )
    r.retrieve_detailed("Hur lång är karenstiden?")
    assert len(vs.calls) == 1                                # exactly one search


def test_subqueries_are_capped():
    vs = FakeVS([_doc("NP-VET-2026", "1. X")])
    many = [{"query": f"q{i}", "scope": "general"} for i in range(6)]
    r = DecomposedRetriever(
        settings=config.get_settings(), vector_store=vs,
        decomposer=_decomp(None, many), max_subqueries=4,
    )
    detailed = r.retrieve_detailed("q")
    assert len(detailed["subqueries"]) == 4
    assert len(vs.calls) == 4


def test_final_set_is_bounded():
    docs = [_doc("NP-VET-2026", f"{i}. Sec{i}") for i in range(20)]
    vs = FakeVS(docs)
    r = DecomposedRetriever(
        settings=config.get_settings(), vector_store=vs,
        decomposer=_decomp(None, [
            {"query": "a", "scope": "general"},
            {"query": "b", "scope": "general"},
            {"query": "c", "scope": "general"},
        ]),
        k_sub=3, max_chunks=6,
    )
    final = r.retrieve_detailed("q")["final"]
    assert len(final) <= 6
