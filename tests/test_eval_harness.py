import config
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from evaluation.harness import DOCS_DIR, RESULT_KEYS, detect_abstention, evaluate_question, load_golden
from generation.prompts import INSUFFICIENT_EVIDENCE_MESSAGE


# ---- Golden dataset validation -------------------------------------------

def test_golden_loads_and_has_records():
    records = load_golden()
    assert len(records) >= 11


def test_golden_ids_unique():
    ids = [r["id"] for r in load_golden()]
    assert len(ids) == len(set(ids))


def test_golden_expected_behavior_counts():
    records = load_golden()
    behaviors = [r["expected_behavior"] for r in records]
    assert set(behaviors) == {"answer", "abstain"}
    # At least some of each. Every record has a valid behavior.
    assert behaviors.count("abstain") >= 1
    assert behaviors.count("answer") >= 1


def test_golden_source_files_exist():
    for record in load_golden():
        for source in record.get("expected_sources", []):
            assert (DOCS_DIR / source["document"]).exists(), source["document"]


# ---- Abstention detection -------------------------------------------------

def test_detects_canonical_abstention_message():
    assert detect_abstention(INSUFFICIENT_EVIDENCE_MESSAGE) is True


def test_does_not_flag_normal_answer():
    assert detect_abstention("Karenstiden för sjukdom är 30 dagar.") is False


# ---- Evaluation result schema (no network; injected retriever + llm) -----

class _FakeRetriever:
    def __init__(self, docs):
        self._docs = docs

    def retrieve(self, question, k=None):
        return self._docs[: (k or len(self._docs))]


def _doc():
    return Document(
        page_content="Karenstiden för sjukdom är 30 dagar.",
        metadata={
            "document_id": "NP-DOG-TERMS-2026",
            "title": "Villkor Hundförsäkring 2026",
            "version": "3.0",
            "effective_date": "2026-01-01",
            "section": "6. Karenstider",
            "source": "dog_insurance_terms_2026.md",
        },
    )


def test_evaluate_question_answer_schema():
    record = {
        "id": "T1",
        "type": "straightforward",
        "question": "Hur lång är karenstiden för sjukdom?",
        "expected_behavior": "answer",
        "expected_sources": [
            {"document": "dog_insurance_terms_2026.md",
             "document_id": "NP-DOG-TERMS-2026",
             "section": "6. Karenstider"},
        ],
    }
    fake_llm = RunnableLambda(lambda _pv: "Karenstiden är 30 dagar.")
    res = evaluate_question(
        record, _FakeRetriever([_doc()]), config.get_settings(), llm=fake_llm
    )

    assert RESULT_KEYS.issubset(res.keys())
    assert res["actual_behavior"] == "answer"
    assert res["retrieved_documents"] == ["NP-DOG-TERMS-2026"]
    assert res["checks"]["expected_source_doc_retrieved"] is True
    assert res["checks"]["expected_section_retrieved"] is True
    assert res["citations"][0]["document_id"] == "NP-DOG-TERMS-2026"


def test_evaluate_question_abstains_when_no_evidence():
    record = {
        "id": "T2",
        "type": "insufficient_evidence",
        "question": "Vad kostar försäkringen?",
        "expected_behavior": "abstain",
        "expected_sources": [],
    }
    res = evaluate_question(record, _FakeRetriever([]), config.get_settings())
    assert res["actual_behavior"] == "abstain"
    assert res["checks"]["behavior_matches_expected"] is True
    assert res["checks"]["expected_source_doc_retrieved"] is None


def test_numeric_question_flagged_for_manual_review():
    record = {
        "id": "T3",
        "type": "careful_interpretation",
        "question": "Hur mycket ersätts?",
        "expected_behavior": "answer",
        "expected_sources": [
            {"document": "cat_insurance_terms_2026.md",
             "document_id": "NP-CAT-TERMS-2026",
             "section": "5. Självrisker"},
        ],
        "calculation": {"result": "6 800 SEK"},
    }
    fake_llm = RunnableLambda(lambda _pv: "Ersättningen blir 6 800 SEK.")
    doc = _doc()
    doc.metadata["document_id"] = "NP-CAT-TERMS-2026"
    doc.metadata["section"] = "5. Självrisker"
    res = evaluate_question(
        record, _FakeRetriever([doc]), config.get_settings(), llm=fake_llm
    )
    assert res["checks"]["is_numeric"] is True
    assert res["checks"]["needs_manual_review"] is True
