import config
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from evaluation.harness import (
    DOCS_DIR,
    RESULT_KEYS,
    canonical_facts,
    deterministic_fact_check,
    detect_abstention,
    evaluate_calculation,
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


def _calc_record(expected_outputs, key_facts, sources):
    return {
        "id": "C",
        "type": "multi_document",
        "question": "q",
        "expected_behavior": "answer",
        "key_facts": key_facts,
        "expected_sources": sources,
        "calculation": {
            "inputs": {},
            "expected_outputs": expected_outputs,
            "reference_steps": [],
        },
    }


_TWO_SOURCES = [
    {"document": "cat_insurance_terms_2026.md",
     "document_id": "NP-CAT-TERMS-2026", "section": "3. Omfattningsnivåer"},
    {"document": "cat_insurance_terms_2026.md",
     "document_id": "NP-CAT-TERMS-2026", "section": "5. Självrisker"},
]


def test_incomplete_retrieval_but_correct_answer_is_ok():
    # Requires 2 pairs but retrieval returns only 1 (incomplete). Answer facts
    # and calculation outputs are all correct -> answer_ok true, no review.
    record = _calc_record(
        {"reimbursement_sek": 6800},
        key_facts=["Endast rörlig självrisk 15 %"],  # numeric -> deterministic
        sources=_TWO_SOURCES,
    )
    fake_llm = RunnableLambda(
        lambda _pv: "Endast rörlig självrisk 15 % tillämpas; ersättningen blir 6 800 SEK."
    )
    doc = _doc(document_id="NP-CAT-TERMS-2026", section="5. Självrisker")
    res = evaluate_question(
        record, _FakeRetriever([doc]), config.get_settings(), llm=fake_llm
    )
    assert res["checks"]["retrieval"]["complete_source_retrieval"] is False
    assert res["checks"]["answer_ok"] is True
    assert res["checks"]["needs_manual_review"] is False


def test_complete_retrieval_but_wrong_calculation_fails():
    record = _calc_record(
        {"reimbursement_sek": 6800},
        key_facts=["Skadan ersätts"],  # prose -> judge (kept supported)
        sources=_TWO_SOURCES,
    )
    fake_llm = RunnableLambda(lambda _pv: "Ersättningen blir 9 999 SEK.")
    docs = [
        _doc(document_id="NP-CAT-TERMS-2026", section="3. Omfattningsnivåer"),
        _doc(document_id="NP-CAT-TERMS-2026", section="5. Självrisker"),
    ]
    res = evaluate_question(
        record, _FakeRetriever(docs), config.get_settings(),
        llm=fake_llm, fact_judge=_support_if("ers"),
    )
    assert res["checks"]["retrieval"]["complete_source_retrieval"] is True
    assert res["checks"]["answer_ok"] is False
    assert res["checks"]["material_contradiction"] is True
    # A wrong calculation is a confident failure, not a review case.
    assert res["checks"]["needs_manual_review"] is False


# ---- Calculation output extraction (layered) ------------------------------

_Q8_OUTPUTS = {"reimbursement_sek": 24225, "customer_out_of_pocket_sek": 5775}
_Q10_OUTPUTS = {"reimbursement_sek": 6800, "customer_out_of_pocket_sek": 1200}


def _calc(outputs):
    return {"expected_behavior": "answer", "calculation": {"expected_outputs": outputs}}


def test_q8_labeled_outputs_incorrect_not_ambiguous():
    # Exact Q8-style generated answer with explicit labels.
    answer = (
        "Ja, förhandsgodkännande krävs. "
        "Ersättning: 30 000 SEK. Kundens självrisk: 6 000 SEK."
    )
    res = evaluate_calculation(_calc(_Q8_OUTPUTS), answer)
    by = {c["field"]: c for c in res["calculation_checks"]}
    assert by["reimbursement_sek"]["actual"] == 30000
    assert by["reimbursement_sek"]["status"] == "incorrect"
    assert by["reimbursement_sek"]["extraction_method"] == "explicit_label"
    assert by["reimbursement_sek"]["evidence"] == "Ersättning: 30 000 SEK"
    assert by["customer_out_of_pocket_sek"]["actual"] == 6000
    assert by["customer_out_of_pocket_sek"]["status"] == "incorrect"
    assert res["ambiguous"] is False
    assert res["all_outputs_correct"] is False


def test_q10_labeled_outputs_incorrect_not_ambiguous():
    answer = "Ersättning: 5 525 SEK. Kundens självrisk: 2 475 SEK."
    res = evaluate_calculation(_calc(_Q10_OUTPUTS), answer)
    by = {c["field"]: c for c in res["calculation_checks"]}
    assert by["reimbursement_sek"]["actual"] == 5525
    assert by["reimbursement_sek"]["status"] == "incorrect"
    assert by["customer_out_of_pocket_sek"]["actual"] == 2475
    assert by["customer_out_of_pocket_sek"]["status"] == "incorrect"
    assert res["ambiguous"] is False


def test_correct_outputs_without_steps():
    answer = "Ersättningen blir 24 225 SEK och kunden betalar 5 775 SEK själv."
    res = evaluate_calculation(_calc(_Q8_OUTPUTS), answer)
    assert res["all_outputs_correct"] is True
    assert res["correct_outputs"] == 2


def test_unrelated_amounts_do_not_disturb_labeled_output():
    # Coverage limit + invoice cost present, but only one labeled reimbursement.
    answer = (
        "Det årliga taket är 120 000 SEK och fakturan var 30 000 SEK. "
        "Ersättning: 24 225 SEK."
    )
    res = evaluate_calculation(_calc({"reimbursement_sek": 24225}), answer)
    c = res["calculation_checks"][0]
    assert c["status"] == "correct"
    assert c["actual"] == 24225
    assert res["ambiguous"] is False


def test_intermediate_values_do_not_disturb_final_deductible():
    # Fixed + variable deductible components appear, but the labeled final is one.
    answer = (
        "Fast självrisk 1 500 SEK dras, sedan 15 % rörlig självrisk 4 275 SEK. "
        "Kundens självrisk: 5 775 SEK."
    )
    res = evaluate_calculation(_calc({"customer_out_of_pocket_sek": 5775}), answer)
    c = res["calculation_checks"][0]
    assert c["status"] == "correct"
    assert c["actual"] == 5775
    assert res["ambiguous"] is False


def test_missing_final_output():
    res = evaluate_calculation(_calc(_Q8_OUTPUTS), "Operationen omfattas av försäkringen.")
    assert all(c["status"] == "missing" for c in res["calculation_checks"])
    assert res["all_outputs_correct"] is False


def test_contradictory_labeled_outputs_are_ambiguous():
    answer = "Ersättning: 5 000 SEK. Efter omräkning: Ersättning: 6 000 SEK."
    res = evaluate_calculation(_calc({"reimbursement_sek": 5500}), answer)
    c = res["calculation_checks"][0]
    assert c["status"] == "ambiguous"
    assert c["actual"] == [5000, 6000]
    assert res["ambiguous"] is True


# ---- Layer-2 LLM extraction fallback (fake extractor, no API) -------------

def test_llm_fallback_used_when_no_explicit_label():
    # No parseable label -> extractor is consulted.
    answer = "Kunden får trettio tusen kronor tillbaka."
    def extractor(_answer):
        return {"reimbursement_sek": 30000, "customer_out_of_pocket_sek": None,
                "confidence": "high", "reason": "stated in words"}
    res = evaluate_calculation(_calc({"reimbursement_sek": 24225}), answer, extractor=extractor)
    c = res["calculation_checks"][0]
    assert c["extraction_method"] == "llm_extraction"
    assert c["actual"] == 30000
    assert c["status"] == "incorrect"


def test_ambiguous_calc_triggers_review():
    record = _calc_record(
        {"reimbursement_sek": 5500}, key_facts=["Skadan ersätts"], sources=_TWO_SOURCES,
    )
    fake_llm = RunnableLambda(
        lambda _pv: "Ersättning: 5 000 SEK. Ersättning: 6 000 SEK."
    )
    res = evaluate_question(
        record, _FakeRetriever([_doc(section="5. Självrisker")]),
        config.get_settings(), llm=fake_llm, fact_judge=_support_if("ers"),
    )
    assert res["checks"]["calculation"]["ambiguous"] is True
    assert res["checks"]["answer_ok"] is False
    assert res["checks"]["needs_manual_review"] is True


def test_low_confidence_llm_extraction_triggers_review():
    record = _calc_record(
        {"reimbursement_sek": 24225}, key_facts=["Skadan ersätts"], sources=_TWO_SOURCES,
    )
    fake_llm = RunnableLambda(lambda _pv: "Kunden får runt tjugofyra tusen tillbaka.")

    def low_conf_extractor(_answer):
        return {"reimbursement_sek": 24225, "customer_out_of_pocket_sek": None,
                "confidence": "low", "reason": "unsure"}

    res = evaluate_question(
        record, _FakeRetriever([_doc(section="5. Självrisker")]),
        config.get_settings(), llm=fake_llm,
        fact_judge=_support_if("tillbaka"), output_extractor=low_conf_extractor,
    )
    assert res["checks"]["answer"]["all_facts_supported"] is True
    assert res["checks"]["needs_manual_review"] is True


def test_high_conf_unsupported_fact_suppresses_low_conf_review():
    # A high-confidence UNsupported fact -> confident failure, so a co-occurring
    # low-confidence signal must NOT force manual review.
    record = _calc_record(
        {"reimbursement_sek": 24225},
        key_facts=["Skadan ersätts inte alls"],  # judged unsupported, high conf
        sources=_TWO_SOURCES,
    )
    fake_llm = RunnableLambda(lambda _pv: "Kunden får runt tjugofyra tusen.")

    def unsupported_high(fact, answer):
        return {"supported": False, "confidence": "high", "reason": "not stated"}

    def low_conf_extractor(_answer):
        return {"reimbursement_sek": 24225, "customer_out_of_pocket_sek": None,
                "confidence": "low", "reason": "unsure"}

    res = evaluate_question(
        record, _FakeRetriever([_doc(section="5. Självrisker")]),
        config.get_settings(), llm=fake_llm,
        fact_judge=unsupported_high, output_extractor=low_conf_extractor,
    )
    assert res["checks"]["answer_ok"] is False          # unsupported fact fails it
    assert res["checks"]["needs_manual_review"] is False  # but confidently, no review


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
