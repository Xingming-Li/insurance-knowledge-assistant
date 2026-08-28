"""Evaluation harness for the Insurance Knowledge Assistant

Runs every question in ``eval/golden_qa.jsonl`` through the retrieval and
generation layers and records what happened. Deterministic retrieval and
answer by design: temperature 0, fixed model config, fixed retrieval-k.

IMPORTANT: the golden dataset is evaluation-only. The ``expected_answer`` is
NEVER shown to the retriever or the generation model — only ``question`` is.

Two quality dimensions are reported SEPARATELY:

* retrieval  — pair-matched (document, section) recall. A same-named section in
               another document does NOT count as a hit. Incomplete retrieval
               does not, by itself, mean the answer is wrong.
* answer     — every ``key_fact`` is evaluated. Numeric/unit-bearing facts are
               checked deterministically; prose facts go to a semantic judge
               that only sees the single fact + the generated answer. The facts
               denominator is always ``len(key_facts)``.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from config import PROJECT_ROOT, Settings, get_settings
from generation.answer import generate_answer
from retrieval.retriever import InsuranceRetriever

GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden_qa.jsonl"
DOCS_DIR = PROJECT_ROOT / "data" / "insurance_docs"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
DEFAULT_RESULTS_PATH = RESULTS_DIR / "latest.json"

# A fact judge is any callable (fact, answer) -> {"supported", "confidence", "reason"}.
FactJudge = Callable[[str, str], Dict[str, Any]]

# Keys guaranteed to be present on every per-question result record.
RESULT_KEYS = {
    "id",
    "type",
    "question",
    "expected_behavior",
    "actual_behavior",
    "retrieved_documents",
    "retrieved_sections",
    "answer",
    "citations",
    "checks",
}

# Deterministic markers that indicate the assistant declared the evidence
# insufficient (either the canonical message or a model-generated equivalent).
_ABSTENTION_MARKERS = (
    "inte tillräcklig",
    "inte räcker",
    "räcker inte",
    "otillräcklig",
    "kan inte besvara",
    "kan tyvärr inte besvara",
    "innehåller ingen information",
    "saknar information",
    "framgår inte av",
    "underlaget räcker inte",
)


def load_golden(path: Path = GOLDEN_PATH) -> List[Dict[str, Any]]:
    """Load the golden Q&A records from JSONL."""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def detect_abstention(answer: str) -> bool:
    """Deterministically decide whether an answer declares insufficient evidence."""
    text = (answer or "").lower()
    return any(marker in text for marker in _ABSTENTION_MARKERS)


# --------------------------------------------------------------------------
# Retrieval quality: source recall (pair-matched)
# --------------------------------------------------------------------------

def source_recall(
    record: Dict[str, Any],
    retrieved_pairs: List[Tuple[Optional[str], Optional[str]]],
) -> Optional[Dict[str, Any]]:
    """Compute (document, section) pair recall for one question.

    Returns None when the question has no expected sources (abstention). A
    section is only counted when both the document AND the section match, so a
    same-named section in a different document is not a false hit.
    """
    expected_sources = record.get("expected_sources", []) or []
    if not expected_sources:
        return None

    required_pairs: Set[Tuple[str, str]] = {
        (s.get("document_id"), s.get("section")) for s in expected_sources
    }
    required_docs: Set[str] = {doc for doc, _ in required_pairs}

    retrieved_set = set(retrieved_pairs)
    retrieved_docs = {doc for doc, _ in retrieved_pairs}

    pairs_hit = required_pairs & retrieved_set
    docs_hit = required_docs & retrieved_docs

    return {
        "docs_recall": f"{len(docs_hit)}/{len(required_docs)}",
        "source_pair_recall": f"{len(pairs_hit)}/{len(required_pairs)}",
        "pairs_retrieved": len(pairs_hit),
        "pairs_required": len(required_pairs),
        "complete_source_retrieval": pairs_hit == required_pairs,
        "missing_pairs": [
            {"document_id": d, "section": s}
            for d, s in sorted(required_pairs - pairs_hit)
        ],
    }


# --------------------------------------------------------------------------
# Answer quality: hybrid key-fact evaluation
# --------------------------------------------------------------------------

_NBSP = " "
_THIN = " "


def canonical_facts(text: str) -> Set[str]:
    """Extract normalized numeric/unit tokens from Swedish insurance text.

    Captures amounts, percentages and durations, normalized so that
    '90 000 SEK', '90 000 kr' and '90000 SEK' collapse to one token.
    """
    if not text:
        return set()
    t = text.replace(_NBSP, " ").replace(_THIN, " ").lower()
    tokens: Set[str] = set()

    for m in re.finditer(r"(\d+)\s*%", t):
        tokens.add(f"{int(m.group(1))}%")

    for m in re.finditer(r"(\d[\d ]*\d|\d)\s*(sek|kr)\b", t):
        tokens.add(f"{int(m.group(1).replace(' ', ''))}sek")

    for m in re.finditer(
        r"(\d+)\s*(dag(?:ar|ars)?|månad(?:er|ers)?|veck(?:a|or)|år)\b", t
    ):
        num, unit = int(m.group(1)), m.group(2)
        if unit.startswith("dag"):
            u = "d"
        elif unit.startswith("månad"):
            u = "mån"
        elif unit.startswith("veck"):
            u = "v"
        else:
            u = "år"
        tokens.add(f"{num}{u}")

    return tokens


def deterministic_fact_check(fact: str, answer: str) -> Optional[bool]:
    """Numeric/unit fact check. Returns None if the fact has no numeric content."""
    fact_tokens = canonical_facts(fact)
    if not fact_tokens:
        return None
    return fact_tokens <= canonical_facts(answer)


# ---- Semantic judge (dependency-injectable) -------------------------------

_JUDGE_SYSTEM = (
    "You are a strict evaluator. You are given exactly ONE key fact and a "
    "generated answer, both in Swedish. Decide whether the answer clearly "
    "expresses the meaning of the key fact. Treat paraphrases and equivalent "
    "wording as supported. Use ONLY the answer text — no outside knowledge or "
    "assumptions. If the answer is silent about the fact, or contradicts it, or "
    "only vaguely alludes to it, mark it unsupported. Return a confidence of "
    "high, medium or low and a short reason."
)
_JUDGE_USER = "KEY FACT:\n{fact}\n\nANSWER:\n{answer}"


class SemanticFactJudge:
    """Default LLM-backed judge. Sees only (fact, answer). Temperature 0.

    Reuses the configured chat model unless ``model`` overrides it.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model: Optional[str] = None,
    ):
        self.settings = settings or get_settings()
        self.model = model or self.settings.judge_model
        self._chain = None

    def _get_chain(self):
        if self._chain is None:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI
            from pydantic import BaseModel, Field

            class FactVerdict(BaseModel):
                supported: bool = Field(description="Does the answer express the key fact?")
                confidence: str = Field(description="high, medium or low")
                reason: str = Field(description="Short justification")

            llm = ChatOpenAI(
                model=self.model,
                temperature=self.settings.judge_temperature,
                api_key=self.settings.require_api_key(),
            )
            prompt = ChatPromptTemplate.from_messages(
                [("system", _JUDGE_SYSTEM), ("human", _JUDGE_USER)]
            )
            self._chain = prompt | llm.with_structured_output(FactVerdict)
        return self._chain

    def __call__(self, fact: str, answer: str) -> Dict[str, Any]:
        verdict = self._get_chain().invoke({"fact": fact, "answer": answer})
        conf = str(verdict.confidence).lower()
        if conf not in ("high", "medium", "low"):
            conf = "low"
        return {
            "supported": bool(verdict.supported),
            "confidence": conf,
            "reason": verdict.reason,
        }


def evaluate_facts(
    record: Dict[str, Any],
    answer: str,
    judge: Optional[FactJudge] = None,
) -> Optional[Dict[str, Any]]:
    """Hybrid per-fact evaluation. Returns None for abstention questions.

    Every key_fact is evaluated; the denominator is always len(key_facts).
    Numeric facts use the deterministic checker; prose facts use ``judge``.
    """
    if record.get("expected_behavior") != "answer":
        return None

    key_facts = record.get("key_facts", []) or []
    fact_checks: List[Dict[str, Any]] = []

    for fact in key_facts:
        det = deterministic_fact_check(fact, answer)
        if det is not None:
            fact_checks.append(
                {
                    "fact": fact,
                    "method": "deterministic",
                    "supported": det,
                    "confidence": "high",
                    "reason": (
                        "numeric/unit tokens present in answer"
                        if det
                        else "expected numeric/unit value not found in answer"
                    ),
                }
            )
        elif judge is not None:
            verdict = judge(fact, answer)
            fact_checks.append(
                {
                    "fact": fact,
                    "method": "semantic_judge",
                    "supported": bool(verdict.get("supported")),
                    "confidence": verdict.get("confidence", "low"),
                    "reason": verdict.get("reason", ""),
                }
            )
        else:
            # No judge available: cannot verify prose fact -> flag low confidence.
            fact_checks.append(
                {
                    "fact": fact,
                    "method": "semantic_judge",
                    "supported": False,
                    "confidence": "low",
                    "reason": "no judge available",
                }
            )

    supported = sum(1 for c in fact_checks if c["supported"])
    total = len(key_facts)
    return {
        "fact_checks": fact_checks,
        "supported_facts": supported,
        "total_facts": total,
        "facts_recall": f"{supported}/{total}",
        "all_facts_supported": total > 0 and supported == total,
    }


# --------------------------------------------------------------------------
# Answer quality: calculation-output evaluation
# --------------------------------------------------------------------------

# Swedish anchor terms that identify which amount in an answer belongs to which
# expected output field. Chosen to avoid false hits (e.g. we anchor
# out-of-pocket on "betalar", NOT "självrisk", so the fixed deductible line does
# not get mistaken for the customer's total cost).
OUTPUT_ANCHORS = {
    "reimbursement_sek": ("ersätt", "återbetal"),
    "customer_out_of_pocket_sek": ("betalar", "egen ficka", "out-of-pocket"),
}


def _amounts_in(clause: str) -> List[int]:
    """SEK amounts found in a single clause."""
    return [
        int(m.group(1).replace(" ", ""))
        for m in re.finditer(r"(\d[\d ]*\d|\d)\s*(sek|kr)\b", clause)
    ]


def _clauses(answer: str) -> List[str]:
    """Lowercase the answer and split into clauses.

    Clause boundaries are strong punctuation and the conjunction 'och', so an
    amount is associated only with an anchor in the SAME clause (e.g.
    'ersättningen blir X' vs 'kunden betalar Y').
    """
    t = answer.replace(_NBSP, " ").replace(_THIN, " ").lower()
    return [c for c in re.split(r"[.;,\n]| och ", t) if c.strip()]


def _extract_output(field: str, expected: int, clauses: List[str]) -> Tuple[Any, str]:
    """Classify the answer's value for one output field.

    status is one of: 'correct', 'incorrect', 'missing', 'ambiguous'.
    """
    all_vals = [v for c in clauses for v in _amounts_in(c)]
    if not all_vals:
        return None, "missing"

    anchors = OUTPUT_ANCHORS.get(field, ())
    associated = [
        v
        for c in clauses
        if anchors and any(a in c for a in anchors)
        for v in _amounts_in(c)
    ]

    if associated:
        distinct = set(associated)
        if expected in distinct:
            return expected, "correct"
        if len(distinct) == 1:
            return associated[0], "incorrect"
        return sorted(distinct), "ambiguous"

    # No clause pins this field: accept an exact match anywhere, else missing.
    if expected in set(all_vals):
        return expected, "correct"
    return None, "missing"


def evaluate_calculation(
    record: Dict[str, Any], answer: str
) -> Optional[Dict[str, Any]]:
    """Evaluate required final calculation OUTPUTS against the answer.

    Only compares final outputs (not reference steps). Returns None when the
    record has no structured ``calculation.expected_outputs``.
    """
    calc = record.get("calculation") or {}
    expected_outputs = calc.get("expected_outputs")
    if not expected_outputs:
        return None

    clauses = _clauses(answer)
    checks = []
    for field, expected in expected_outputs.items():
        actual, status = _extract_output(field, expected, clauses)
        checks.append(
            {"field": field, "expected": expected, "actual": actual, "status": status}
        )

    correct = sum(1 for c in checks if c["status"] == "correct")
    total = len(checks)
    return {
        "calculation_checks": checks,
        "correct_outputs": correct,
        "total_outputs": total,
        "all_outputs_correct": total > 0 and correct == total,
        "ambiguous": any(c["status"] == "ambiguous" for c in checks),
    }


# --------------------------------------------------------------------------
# Per-question evaluation
# --------------------------------------------------------------------------

def evaluate_question(
    record: Dict[str, Any],
    retriever: Any,
    settings: Settings,
    llm: Any = None,
    fact_judge: Optional[FactJudge] = None,
) -> Dict[str, Any]:
    """Evaluate a single golden question. Only ``question`` is fed downstream."""
    question = record["question"]

    docs = retriever.retrieve(question, k=settings.retrieval_k)
    result = generate_answer(question, docs, settings=settings, llm=llm)

    retrieved_documents = [(d.metadata or {}).get("document_id") for d in docs]
    retrieved_sections = [(d.metadata or {}).get("section") for d in docs]
    retrieved_pairs = list(zip(retrieved_documents, retrieved_sections))

    abstained = detect_abstention(result.answer) or not result.used_evidence
    actual_behavior = "abstain" if abstained else "answer"
    behavior_matches = actual_behavior == record["expected_behavior"]

    # Retrieval quality is a SEPARATE diagnostic; it never feeds answer_ok.
    retrieval = source_recall(record, retrieved_pairs)

    # Answer quality: key facts + calculation outputs, evaluated independently.
    answer_quality = evaluate_facts(record, result.answer, judge=fact_judge)
    calculation = evaluate_calculation(record, result.answer)

    facts_ok = bool(answer_quality and answer_quality["all_facts_supported"])
    calc_ok = calculation is None or calculation["all_outputs_correct"]
    material_contradiction = bool(
        calculation
        and any(c["status"] == "incorrect" for c in calculation["calculation_checks"])
    )

    if record["expected_behavior"] == "answer":
        answer_ok = (
            behavior_matches and facts_ok and calc_ok and not material_contradiction
        )
    else:
        answer_ok = behavior_matches

    # Manual review ONLY when the evaluator cannot confidently decide:
    # low-confidence semantic judgment or ambiguous numeric extraction.
    # Wrong calculations, missing facts and incomplete retrieval are confident
    # automatic failures, NOT review cases.
    low_confidence = bool(
        answer_quality
        and any(
            c["method"] == "semantic_judge" and c["confidence"] == "low"
            for c in answer_quality["fact_checks"]
        )
    )
    ambiguous_calc = bool(calculation and calculation["ambiguous"])
    needs_manual_review = low_confidence or ambiguous_calc

    return {
        "id": record["id"],
        "type": record.get("type"),
        "question": question,
        "expected_behavior": record["expected_behavior"],
        "actual_behavior": actual_behavior,
        "retrieved_documents": retrieved_documents,
        "retrieved_sections": retrieved_sections,
        "answer": result.answer,
        "citations": [asdict(c) for c in result.citations],
        "checks": {
            "behavior_matches_expected": behavior_matches,
            "abstained": abstained,
            "is_numeric": "calculation" in record,
            "answer_ok": answer_ok,
            "material_contradiction": material_contradiction,
            "retrieval": retrieval,
            "answer": answer_quality,
            "calculation": calculation,
            "needs_manual_review": needs_manual_review,
        },
    }


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    answerable = [r for r in results if r["expected_behavior"] == "answer"]
    abstain = [r for r in results if r["expected_behavior"] == "abstain"]

    retr = [r["checks"]["retrieval"] for r in answerable if r["checks"]["retrieval"]]
    complete = sum(1 for rc in retr if rc["complete_source_retrieval"])
    pairs_hit = sum(rc["pairs_retrieved"] for rc in retr)
    pairs_req = sum(rc["pairs_required"] for rc in retr)

    ans = [r["checks"]["answer"] for r in answerable if r["checks"]["answer"]]
    facts_sup = sum(a["supported_facts"] for a in ans)
    facts_tot = sum(a["total_facts"] for a in ans)
    all_supported = sum(1 for a in ans if a["all_facts_supported"])

    calcs = [r["checks"]["calculation"] for r in answerable if r["checks"]["calculation"]]
    calc_correct = sum(1 for c in calcs if c["all_outputs_correct"])
    answer_ok_count = sum(1 for r in answerable if r["checks"]["answer_ok"])

    return {
        "n_questions": len(results),
        "behavior_match": sum(
            1 for r in results if r["checks"]["behavior_matches_expected"]
        ),
        "retrieval": {
            "complete_source_retrieval": f"{complete}/{len(retr)}",
            "source_pair_recall": f"{pairs_hit}/{pairs_req}",
        },
        "answer": {
            "answered": sum(1 for r in answerable if r["actual_behavior"] == "answer"),
            "answerable": len(answerable),
            "answer_ok": f"{answer_ok_count}/{len(answerable)}",
            "key_fact_recall": f"{facts_sup}/{facts_tot}",
            "all_facts_supported": f"{all_supported}/{len(ans)}",
            "calculation_correct": f"{calc_correct}/{len(calcs)}",
        },
        "abstain": {
            "count": len(abstain),
            "correctly_abstained": sum(
                1 for r in abstain if r["actual_behavior"] == "abstain"
            ),
        },
        "needs_manual_review": [
            r["id"] for r in results if r["checks"]["needs_manual_review"]
        ],
    }


def run_evaluation(
    settings: Optional[Settings] = None,
    retriever: Any = None,
    llm: Any = None,
    fact_judge: Optional[FactJudge] = None,
    golden_path: Path = GOLDEN_PATH,
) -> Dict[str, Any]:
    """Run the full evaluation and return a machine-readable payload."""
    settings = settings or get_settings()
    retriever = retriever or InsuranceRetriever(settings=settings)
    if fact_judge is None:
        fact_judge = SemanticFactJudge(settings=settings)
    records = load_golden(golden_path)

    results = [
        evaluate_question(r, retriever, settings, llm=llm, fact_judge=fact_judge)
        for r in records
    ]

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "temperature": settings.chat_temperature,
        "retrieval_k": settings.retrieval_k,
        "collection_name": settings.collection_name,
    }
    return {"meta": meta, "summary": _summarize(results), "results": results}


def save_results(payload: Dict[str, Any], path: Path = DEFAULT_RESULTS_PATH) -> Path:
    """Persist the evaluation payload as JSON. Returns the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def print_report(payload: Dict[str, Any]) -> None:
    """Human-readable console report (retrieval and answer quality separated)."""
    meta = payload["meta"]
    summary = payload["summary"]

    print("NordicPaws — Evaluation report")
    print(
        f"model={meta['chat_model']}  emb={meta['embedding_model']}  "
        f"temp={meta['temperature']}  k={meta['retrieval_k']}"
    )
    print("=" * 96)
    print(
        f"{'QID':<4} {'exp':<8} {'act':<8} {'pairs':<7} {'facts':<7} {'calc':<6} "
        f"{'answer_ok':<10} {'review':<6}"
    )
    print("-" * 96)
    for r in payload["results"]:
        c = r["checks"]
        pairs = c["retrieval"]["source_pair_recall"] if c["retrieval"] else "-"
        facts = c["answer"]["facts_recall"] if c["answer"] else "-"
        cc = c["calculation"]
        calc = f"{cc['correct_outputs']}/{cc['total_outputs']}" if cc else "-"
        answer_ok = "yes" if c["answer_ok"] else "NO"
        print(
            f"{r['id']:<4} {r['expected_behavior']:<8} {r['actual_behavior']:<8} "
            f"{pairs:<7} {facts:<7} {calc:<6} {answer_ok:<10} "
            f"{('YES' if c['needs_manual_review'] else '-'):<6}"
        )

    print("=" * 92)
    rt, an, ab = summary["retrieval"], summary["answer"], summary["abstain"]
    print(
        f"BEHAVIOR MATCH: {summary['behavior_match']}/{summary['n_questions']}   "
        f"answered: {an['answered']}/{an['answerable']}   "
        f"abstained: {ab['correctly_abstained']}/{ab['count']}"
    )
    print(
        f"RETRIEVAL  all-source-retrieved: {rt['complete_source_retrieval']}   "
        f"pair recall: {rt['source_pair_recall']}"
    )
    print(
        f"ANSWER     answer_ok: {an['answer_ok']}   "
        f"all-facts-supported: {an['all_facts_supported']}   "
        f"key-fact recall: {an['key_fact_recall']}   "
        f"calc-correct: {an['calculation_correct']}"
    )
    print(f"NEEDS MANUAL REVIEW: {', '.join(summary['needs_manual_review']) or 'none'}")
