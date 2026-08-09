import config
from langchain_core.documents import Document

from retrieval.retriever import InsuranceRetriever


class FakeVectorStore:
    """Stand-in for Chroma so retrieval logic is testable without network."""

    def __init__(self, docs):
        self._docs = docs
        self.last_k = None

    def similarity_search(self, question, k):
        self.last_k = k
        return self._docs[:k]


def _doc(i):
    return Document(
        page_content=f"chunk {i}",
        metadata={
            "document_id": "NP-DOG-TERMS-2026",
            "title": "Villkor Hundförsäkring 2026",
            "version": "3.0",
            "section": "6. Karenstider",
            "source": "dog_insurance_terms_2026.md",
        },
    )


def test_retrieve_respects_k_and_returns_source_metadata():
    fake = FakeVectorStore([_doc(i) for i in range(5)])
    retriever = InsuranceRetriever(settings=config.get_settings(), vector_store=fake)

    results = retriever.retrieve("Hur lång är karenstiden?", k=3)

    assert len(results) == 3
    assert fake.last_k == 3
    meta = results[0].metadata
    assert meta["document_id"] == "NP-DOG-TERMS-2026"
    assert meta["section"] == "6. Karenstider"
    assert meta["version"] == "3.0"


def test_retrieve_uses_configured_default_k():
    fake = FakeVectorStore([_doc(i) for i in range(10)])
    retriever = InsuranceRetriever(settings=config.get_settings(), vector_store=fake)
    retriever.retrieve("fråga")
    assert fake.last_k == config.get_settings().retrieval_k
