import config
from langchain_core.documents import Document

from retrieval.retriever import (
    CAT_TERMS_DOC,
    DOG_TERMS_DOC,
    InsuranceRetriever,
    detect_species,
    species_filter,
)

GENERAL_DOCS = ["NP-VET-2026", "NP-CLAIMS-2026", "NP-EXCL-2026", "NP-CS-2026"]


def _passes(document_id, flt):
    if not flt:
        return True
    nin = flt["document_id"]["$nin"]
    return document_id not in nin


class FakeVS:
    """Records the filter and applies it (like Chroma's where)."""

    def __init__(self, doc_ids):
        self.docs = [Document(page_content="x", metadata={"document_id": d, "section": "1. X"}) for d in doc_ids]
        self.last_filter = "unset"

    def similarity_search(self, question, k, filter=None):
        self.last_filter = filter
        return [d for d in self.docs if _passes(d.metadata["document_id"], filter)][:k]


ALL_DOCS = [DOG_TERMS_DOC, CAT_TERMS_DOC] + GENERAL_DOCS


# ---- detect_species (deterministic, lexical) ------------------------------

def test_detects_dog_forms():
    for q in ["Min hund har Premium", "hunden behöver vård", "två hundar", "hundförsäkring"]:
        assert detect_species(q) == "dog", q


def test_detects_cat_forms():
    for q in ["Min katt är sjuk", "katten behöver vård", "flera katter", "kattförsäkring"]:
        assert detect_species(q) == "cat", q


def test_ambiguous_and_unknown_return_none():
    assert detect_species("Jag har både hund och katt") is None
    assert detect_species("Vad är karenstiden?") is None
    assert detect_species("Det kostar hundra kronor") is None  # 'hundra' != 'hund'


# ---- retrieval filtering --------------------------------------------------

def test_dog_question_excludes_cat_terms_but_keeps_general():
    vs = FakeVS(ALL_DOCS)
    r = InsuranceRetriever(settings=config.get_settings(), vector_store=vs)
    got = {(d.metadata["document_id"]) for d in r.retrieve("Min hund behöver operation", k=8)}
    assert CAT_TERMS_DOC not in got
    assert DOG_TERMS_DOC in got
    for g in GENERAL_DOCS:
        assert g in got


def test_cat_question_excludes_dog_terms_but_keeps_general():
    vs = FakeVS(ALL_DOCS)
    r = InsuranceRetriever(settings=config.get_settings(), vector_store=vs)
    got = {(d.metadata["document_id"]) for d in r.retrieve("Min katt behöver tandvård", k=8)}
    assert DOG_TERMS_DOC not in got
    assert CAT_TERMS_DOC in got
    for g in GENERAL_DOCS:
        assert g in got


def test_unknown_species_applies_no_filter():
    vs = FakeVS(ALL_DOCS)
    r = InsuranceRetriever(settings=config.get_settings(), vector_store=vs)
    got = {(d.metadata["document_id"]) for d in r.retrieve("Vad är karenstiden?", k=8)}
    assert vs.last_filter is None
    assert set(ALL_DOCS) == got  # nothing excluded


def test_species_filter_shapes():
    assert species_filter("min hund") == {"document_id": {"$nin": [CAT_TERMS_DOC]}}
    assert species_filter("min katt") == {"document_id": {"$nin": [DOG_TERMS_DOC]}}
    assert species_filter("ingen art här") is None
