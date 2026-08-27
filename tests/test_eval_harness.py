import config
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from evaluation.harness import (
    DOCS_DIR,
    RESULT_KEYS,
    canonical_facts,
    deterministic_fact_check,
    detect_abstention,
    evaluate_facts,
    evaluate_question,
    load_golden,
    source_recall,
)
from generation.prompts import INSUFFICIENT_EVIDENCE_MESSAGE


# ---- Golden dataset validation -------------------------------------------

def test_golden_loads_and_has_records():
    assert len(load_golden()) >= 11


def test_golden_ids_unique():
    ids = [r["id"] for r in load_golden()]
    assert len(ids) == len(set(ids))


def test_golden_expected_behavior_counts():
    behaviors = [r["expected_behavior"] for r in load_golden()]
    assert set(behaviors) == {"answer", "abstain"}
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
    retrieved = [("NP-A", "1. Alpha"), ("NP-A", "2. Beta"), ("NP-C", "9. Other")]
    rc = source_recall(record, retrieved)
    assert rc["source_pair_recall"] == "2/3"
    assert rc["docs_recall"] == "1/2"
    assert rc["complete_source_retrieval"] is False
    assert {"document_id": "NP-B", "section": "3. Gamma"} in rc["missing_pairs"]

    rc2 = source_recall(
        record, [("NP-A", "1. Alpha"), ("NP-A", "2. Beta"), ("NP-B", "3. Gamma")]
    )
    assert rc2["source_pair_recall"] == "3/3"
    assert rc2["complete_source_retrieval"] is True


def test_source_recall_rejects_section_name_collision():
    record = _record_with_sources([("NP-CAT-TERMS-2026", "3. Omfattningsnivåer")])
    retrieved = [("NP-DOG-TERMS-2026", "3. Omfattningsnivåer")]
    rc = source_recall(record, retrieved)
    assert rc["source_pair_recall"] == "0/1"
    assert rc["complete_source_retrieval"] is False


# ---- Deterministic fact checks -------------------------------------------

def test_canonical_facts_normalizes_units():
    toks = canonical_facts("90 000 SEK, 15 %, 30 dagar, 12 månader, 1 200 kr")
    assert {"90000sek", "15%", "30d", "12mån", "1200sek"} <= toks


def test_deterministic_fact_pass_and_fail():
    assert deterministic_fact_check("rörlig självrisk 15 %", "... på 15 % ...") is True
    assert deterministic_fact_check("taket är 90 000 SEK", "taket är 80 000 SEK") is False
    # No numeric content -> not deterministically checkable.
    assert deterministic_fact_check("Olycksfall: ingen karens", "text") is None


# ---- Hybrid fact evaluation (with a fake, injectable judge) ---------------

def _support_if(*needles):
    def judge(fact, answer):
        ok = any(n.lower() in answer.lower() for n in needles)
        return {"supported": ok, "confidence": "high", "reason": "fake"}
    return judge


def test_denominator_always_equals_len_key_facts():
    record = {
        "expected_behavior": "answer",
        "key_facts": [
            "rörlig självrisk 15 %",                 # numeric -> deterministic
            "Ersättningen blir 6 800 SEK",           # numeric -> deterministic
            "Fast självrisk redan dragen i perioden",  # prose -> judge
        ],
    }
    answer = "Endast rörlig 15 % tillämpas; ersättningen blir 6 800 SEK; fast självrisk redan dragen."
    res = evaluate_facts(record, answer, judge=_support_if("redan dragen"))
    assert res["total_facts"] == 3
    assert res["facts_recall"] == "3/3"
    assert res["all_facts_supported"] is True
    methods = {c["method"] for c in res["fact_checks"]}
    assert methods == {"deterministic", "semantic_judge"}


def test_paraphrase_supported_and_missing_unsupported():
    record = {
        "expected_behavior": "answer",
        "key_facts": ["Akut veterinärvård kräver inget förhandsgodkännande."],
    }
    supported = evaluate_facts(
        record,
        "Vid akut behandling behövs inget förhandsgodkännande.",
        judge=_support_if("förhandsgodkännande"),
    )
    assert supported["all_facts_supported"] is True
    assert supported["fact_checks"][0]["method"] == "semantic_judge"

    missing = evaluate_facts(
        record,
        "Skadan ska anmälas så snart som möjligt.",
        judge=_support_if("NON-MATCH"),
    )
    assert missing["all_facts_supported"] is False


def test_evaluate_facts_none_for_abstain():
    record = {"expected_behavior": "abstain", "key_facts": ["ingen prisinformation"]}
    assert evaluate_facts(record, "Underlaget räcker inte.", judge=_support_if("x")) is None


# ---- evaluate_question: schema + manual-review policy ---------------------

class _FakeRetriever:
    def __init__(self, docs):
        self._docs = docs

    def retrieve(self, question, k=None):
        return self._docs[: (k or len(self._docs))]


def _doc(document_id="NP-DOG-TERMS-2026", section="6. Karenstider", content="text"):
    return Document(
        page_content=content,
        metadata={
            "document_id": document_id,
            "title": "T",
            "version": "3.0",
            "effective_date": "2026-01-01",
            "section": section,
            "source": "f.md",
        },
    )


def test_evaluate_question_schema_and_answer_ok():
    record = {
        "id": "T1",
        "type": "straightforward",
        "question": "Hur lång är karenstiden för sjukdom?",
        "expected_behavior": "answer",
        "key_facts": ["Generell sjukdom: 30 dagars karens"],
        "expected_sources": [
            {"document": "dog_insurance_terms_2026.md",
             "document_id": "NP-DOG-TERMS-2026", "section": "6. Karenstider"},
        ],
    }
    fake_llm = RunnableLambda(lambda _pv: "Karenstiden för sjukdom är 30 dagar.")
    res = evaluate_question(
        record, _FakeRetriever([_doc()]), config.get_settings(), llm=fake_llm
    )
    assert RESULT_KEYS.issubset(res.keys())
    assert res["checks"]["retrieval"]["source_pair_recall"] == "1/1"
    assert res["checks"]["answer"]["facts_recall"] == "1/1"
    assert res["checks"]["answer_ok"] is True
    assert res["checks"]["needs_manual_review"] is False


def test_low_confidence_triggers_manual_review():
    record = {
        "id": "T2",
        "type": "straightforward",
        "question": "Krävs förhandsgodkännande akut?",
        "expected_behavior": "answer",
        "key_facts": ["Akut vård kräver inget förhandsgodkännande."],
        "expected_sources": [
            {"document": "veterinary_care_guidelines_2026.md",
             "document_id": "NP-VET-2026", "section": "2. Akut- och jourvård"},
        ],
    }
    fake_llm = RunnableLambda(lambda _pv: "Inget förhandsgodkännande vid akut vård.")

    def low_conf_judge(fact, answer):
        return {"supported": True, "confidence": "low", "reason": "unsure"}

    res = evaluate_question(
        record,
        _FakeRetriever([_doc(document_id="NP-VET-2026", section="2. Akut- och jourvård")]),
        config.get_settings(),
        llm=fake_llm,
        fact_judge=low_conf_judge,
    )
    # All facts nominally supported, but low judge confidence forces review.
    assert res["checks"]["answer"]["all_facts_supported"] is True
    assert res["checks"]["needs_manual_review"] is True


def test_incomplete_retrieval_no_review_when_facts_supported():
    # Requires 2 pairs but retrieval returns only 1, and it's a numeric question.
    record = {
        "id": "T3",
        "type": "careful_interpretation",
        "question": "Hur mycket ersätts?",
        "expected_behavior": "answer",
        "key_facts": ["Ersättningen blir 6 800 SEK"],  # numeric -> deterministic
        "expected_sources": [
            {"document": "cat_insurance_terms_2026.md",
             "document_id": "NP-CAT-TERMS-2026", "section": "3. Omfattningsnivåer"},
            {"document": "cat_insurance_terms_2026.md",
             "document_id": "NP-CAT-TERMS-2026", "section": "5. Självrisker"},
        ],
        "calculation": {"result": "6 800 SEK"},
    }
    fake_llm = RunnableLambda(lambda _pv: "Ersättningen blir 6 800 SEK.")
    # Only one of the two required pairs retrieved -> incomplete retrieval.
    doc = _doc(document_id="NP-CAT-TERMS-2026", section="5. Självrisker")
    res = evaluate_question(
        record, _FakeRetriever([doc]), config.get_settings(), llm=fake_llm
    )
    assert res["checks"]["retrieval"]["complete_source_retrieval"] is False
    assert res["checks"]["is_numeric"] is True
    assert res["checks"]["answer"]["all_facts_supported"] is True
    # Incomplete retrieval + numeric question must NOT force review here.
    assert res["checks"]["needs_manual_review"] is False


def test_abstention_bypasses_fact_judging():
    called = {"n": 0}

    def counting_judge(fact, answer):
        called["n"] += 1
        return {"supported": True, "confidence": "high", "reason": "x"}

    record = {
        "id": "T4",
        "type": "insufficient_evidence",
        "question": "Vad kostar försäkringen?",
        "expected_behavior": "abstain",
        "key_facts": ["Ingen prisinformation finns"],
        "expected_sources": [],
    }
    res = evaluate_question(
        record, _FakeRetriever([]), config.get_settings(), fact_judge=counting_judge
    )
    assert res["actual_behavior"] == "abstain"
    assert res["checks"]["answer"] is None
    assert res["checks"]["retrieval"] is None
    assert res["checks"]["answer_ok"] is True
    assert called["n"] == 0  # judge never invoked for abstention
