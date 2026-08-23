import config
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from evaluation.harness import (
    DOCS_DIR,
    RESULT_KEYS,
    answer_correctness,
    canonical_facts,
    detect_abstention,
    evaluate_question,
    load_golden,
    source_recall,
)
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
    rc = res["checks"]["source_recall"]
    assert rc["pairs_recall"] == "1/1"
    assert rc["complete"] is True
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
    assert res["checks"]["source_recall"] is None


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


# ---- Source-pair recall ---------------------------------------------------

def _record_with_sources(pairs):
    return {
        "id": "S",
        "type": "multi_document",
        "expected_behavior": "answer",
        "expected_sources": [
            {"document": "x.md", "document_id": d, "section": s} for d, s in pairs
        ],
    }


def test_source_recall_partial_and_complete():
    record = _record_with_sources(
        [("NP-A", "1. Alpha"), ("NP-A", "2. Beta"), ("NP-B", "3. Gamma")]
    )
    # Retrieved only 2 of the 3 required pairs (missing NP-B/3. Gamma).
    retrieved = [("NP-A", "1. Alpha"), ("NP-A", "2. Beta"), ("NP-C", "9. Other")]
    rc = source_recall(record, retrieved)
    assert rc["pairs_recall"] == "2/3"
    assert rc["docs_recall"] == "1/2"
    assert rc["complete"] is False
    assert {"document_id": "NP-B", "section": "3. Gamma"} in rc["missing_pairs"]

    retrieved_all = [("NP-A", "1. Alpha"), ("NP-A", "2. Beta"), ("NP-B", "3. Gamma")]
    rc2 = source_recall(record, retrieved_all)
    assert rc2["pairs_recall"] == "3/3"
    assert rc2["complete"] is True


def test_source_recall_rejects_section_name_collision():
    # Expected Cat §3; a Dog §3 with the SAME section name must NOT count.
    record = _record_with_sources([("NP-CAT-TERMS-2026", "3. Omfattningsnivåer")])
    retrieved = [("NP-DOG-TERMS-2026", "3. Omfattningsnivåer")]
    rc = source_recall(record, retrieved)
    assert rc["pairs_recall"] == "0/1"
    assert rc["complete"] is False


# ---- Answer correctness (deterministic numeric matching) ------------------

def test_canonical_facts_normalizes_units():
    toks = canonical_facts("90 000 SEK, 15 %, 30 dagar, 12 månader, 1 200 kr")
    assert {"90000sek", "15%", "30d", "12mån", "1200sek"} <= toks


def test_answer_correctness_supported_and_unsupported():
    record = {
        "expected_behavior": "answer",
        "key_facts": [
            "Cat Plus: rörlig självrisk 15 %",       # numeric -> checkable
            "Ersättningen blir 6 800 SEK",           # numeric -> checkable
            "Fast självrisk redan dragen i perioden",  # no number -> uncheckable
        ],
    }
    answer = "Endast rörlig självrisk på 15 % tillämpas; ersättningen blir 6 800 SEK."
    co = answer_correctness(record, answer)
    assert co["facts_recall"] == "2/2"
    assert co["key_facts_supported"] == 2
    assert len(co["uncheckable_facts"]) == 1

    wrong = "Ersättningen blir 9 999 SEK."
    co2 = answer_correctness(record, wrong)
    assert co2["key_facts_supported"] == 0
    assert len(co2["unsupported_facts"]) == 2


def test_answer_correctness_none_for_abstain():
    record = {"expected_behavior": "abstain", "key_facts": ["ingen prisinformation"]}
    assert answer_correctness(record, "Underlaget räcker inte.") is None
