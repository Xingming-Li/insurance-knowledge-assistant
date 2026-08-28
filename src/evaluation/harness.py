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

# A calc-output extractor is any callable (answer) -> {"reimbursement_sek", ...,
# "confidence", "reason"} reporting only the FINAL amounts the answer claims.
CalcExtractor = Callable[[str], Dict[str, Any]]

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


_EXTRACTOR_SYSTEM = (
    "You extract the FINAL monetary amounts that a Swedish pet-insurance answer "
    "claims. Report the reimbursement the customer receives (ersättning) and the "
    "customer's total out-of-pocket cost / self-risk (kundens självrisk / vad "
    "kunden betalar), each as an integer number of SEK, or null if the answer "
    "does not state it. Report ONLY what the answer claims — do NOT compute, "
    "correct or judge anything, and you are given no correct values. Ignore "
    "intermediate amounts such as invoice cost, coverage limits and the "
    "fixed/variable deductible components; report the final totals only. Give a "
    "confidence of high, medium or low and a short reason."
)


class CalculationOutputExtractor:
    """LLM fallback that reports the final amounts an answer claims.

    Sees only the answer; never the golden values. Reuses the judge model.
    """

    def __init__(self, settings: Optional[Settings] = None, model: Optional[str] = None):
        self.settings = settings or get_settings()
        self.model = model or self.settings.judge_model
        self._chain = None

    def _get_chain(self):
        if self._chain is None:
            from typing import Optional as Opt

            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI
            from pydantic import BaseModel, Field

            class OutputClaims(BaseModel):
                reimbursement_sek: Opt[int] = Field(
                    None, description="Final reimbursement amount claimed (SEK), or null"
                )
                customer_out_of_pocket_sek: Opt[int] = Field(
                    None, description="Final customer out-of-pocket/self-risk (SEK), or null"
                )
                confidence: str = Field(description="high, medium or low")
                reason: str = Field(description="Short justification")

            llm = ChatOpenAI(
                model=self.model,
                temperature=0,
                api_key=self.settings.require_api_key(),
            )
            prompt = ChatPromptTemplate.from_messages(
                [("system", _EXTRACTOR_SYSTEM), ("human", "{answer}")]
            )
            self._chain = prompt | llm.with_structured_output(OutputClaims)
        return self._chain

    def __call__(self, answer: str) -> Dict[str, Any]:
        claims = self._get_chain().invoke({"answer": answer})
        conf = str(claims.confidence).lower()
        if conf not in ("high", "medium", "low"):
            conf = "low"
        return {
            "reimbursement_sek": claims.reimbursement_sek,
            "customer_out_of_pocket_sek": claims.customer_out_of_pocket_sek,
            "confidence": conf,
            "reason": claims.reason,
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

# Swedish label patterns for explicitly stated FINAL outputs. The amount is the
# named group ``val``. Matched with highest priority so clearly labeled outputs
# are never made ambiguous by intermediate arithmetic, invoice cost or coverage
# limits. A label needs an explicit connector (":"/"är"/"blir"/…), so a bare
# "fast självrisk 1 500 SEK" line is NOT treated as a final output.
_V = r"(?P<val>\d[\d ]*\d|\d)\s*(?:sek|kr)"
_OUTPUT_PATTERNS = {
    "reimbursement_sek": [
        r"ersättning(?:en|sbelopp)?(?:\s+som\s+kunden\s+får)?\s*(?::|är|blir|uppgår till|landar på)\s*" + _V,
        _V + r"\s+i\s+ersättning",
    ],
    "customer_out_of_pocket_sek": [
        r"(?:kundens?\s+)?(?:total\s+)?självrisk(?:en)?\s*(?::|är|blir|uppgår till)\s*" + _V,
        r"kundens?\s+egen\s+kostnad\s*(?::|är|blir)?\s*" + _V,
        r"totalt\s+blir\s+kundens\s+självrisk\s*" + _V,
        r"kunden\s+betalar\s+" + _V,
    ],
}


def _explicit_output_matches(
    field: str, text_l: str, orig: str, summary_idx: Optional[int]
) -> List[Tuple[int, str]]:
    """Return [(value, evidence), ...] for explicitly labeled outputs of a field.

    If a 'Sammanfattning' section exists and holds labeled outputs, only those
    are used (a final summary beats intermediate mentions).
    """
    found = []  # (value, evidence, start)
    for pat in _OUTPUT_PATTERNS.get(field, ()):
        for m in re.finditer(pat, text_l):
            value = int(m.group("val").replace(" ", ""))
            found.append((value, orig[m.start():m.end()].strip(), m.start()))

    if summary_idx is not None:
        in_summary = [f for f in found if f[2] >= summary_idx]
        if in_summary:
            found = in_summary

    seen, out = set(), []
    for value, evidence, _ in found:
        if (value, evidence) not in seen:
            seen.add((value, evidence))
            out.append((value, evidence))
    return out


def _classify_output(field, expected, value, method, evidence, confidence, status=None):
    return {
        "field": field,
        "expected": expected,
        "actual": value,
        "status": status or ("correct" if value == expected else "incorrect"),
        "extraction_method": method,
        "evidence": evidence,
        "confidence": confidence,
    }


def evaluate_calculation(
    record: Dict[str, Any],
    answer: str,
    extractor: Optional[CalcExtractor] = None,
) -> Optional[Dict[str, Any]]:
    """Layered extraction of FINAL calculation outputs, compared to golden.

    Layer 1 (deterministic): explicit Swedish output labels.
    Layer 2 (fallback): a semantic LLM extractor that only reports which final
    amounts the answer claims (never sees golden, never judges correctness).
    Only final outputs are compared — reference steps are ignored.

    status per field: 'correct' | 'incorrect' | 'missing' | 'ambiguous'.
    """
    calc = record.get("calculation") or {}
    expected_outputs = calc.get("expected_outputs")
    if not expected_outputs:
        return None

    orig = answer.replace(_NBSP, " ").replace(_THIN, " ")
    text_l = orig.lower()
    m = re.search(r"sammanfattning", text_l)
    summary_idx = m.start() if m else None

    resolved: Dict[str, Dict[str, Any]] = {}
    unresolved: List[str] = []
    for field, expected in expected_outputs.items():
        matches = _explicit_output_matches(field, text_l, orig, summary_idx)
        if not matches:
            unresolved.append(field)
            continue
        distinct = sorted({v for v, _ in matches})
        if len(distinct) == 1:
            resolved[field] = _classify_output(
                field, expected, distinct[0], "explicit_label", matches[0][1], "high"
            )
        else:
            resolved[field] = _classify_output(
                field, expected, distinct, "explicit_label",
                [e for _, e in matches], "high", status="ambiguous",
            )

    # Layer 2: semantic fallback only for fields Layer 1 could not resolve.
    llm_result = extractor(answer) if (unresolved and extractor is not None) else None
    for field in unresolved:
        expected = expected_outputs[field]
        claimed = llm_result.get(field) if llm_result else None
        if claimed is not None:
            resolved[field] = _classify_output(
                field, expected, claimed, "llm_extraction",
                (llm_result or {}).get("reason", ""),
                (llm_result or {}).get("confidence", "low"),
            )
        else:
            resolved[field] = _classify_output(
                field, expected, None,
                "llm_extraction" if llm_result is not None else "none",
                (llm_result or {}).get("reason") if llm_result else None,
                (llm_result or {}).get("confidence", "low") if llm_result else "n/a",
                status="missing",
            )

    checks = [resolved[f] for f in expected_outputs]
    correct = sum(1 for c in checks if c["status"] == "correct")
    total = len(checks)
    return {
        "calculation_checks": checks,
        "correct_outputs": correct,
        "total_outputs": total,
        "all_outputs_correct": total > 0 and correct == total,
        "ambiguous": any(c["status"] == "ambiguous" for c in checks),
        "low_confidence": any(
            c["extraction_method"] == "llm_extraction" and c["confidence"] == "low"
            for c in checks
        ),
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
    output_extractor: Optional[CalcExtractor] = None,
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
    calculation = evaluate_calculation(
        record, result.answer, extractor=output_extractor
    )

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
    # ambiguous numeric extraction, or a low-confidence judgment/extraction.
    # Wrong calculations, missing facts and incomplete retrieval are confident
    # automatic failures, NOT review cases. Also, a high-confidence UNsupported
    # fact means we are already confident the answer is wrong, so low-confidence
    # signals elsewhere should not force a review.
    fact_checks = answer_quality["fact_checks"] if answer_quality else []
    low_conf_fact = any(
        c["method"] == "semantic_judge" and c["confidence"] == "low"
        for c in fact_checks
    )
    high_conf_unsupported = any(
        c["method"] == "semantic_judge" and not c["supported"] and c["confidence"] == "high"
        for c in fact_checks
    )
    ambiguous_calc = bool(calculation and calculation["ambiguous"])
    low_conf_calc = bool(calculation and calculation["low_confidence"])
    low_confidence = (low_conf_fact or low_conf_calc) and not high_conf_unsupported
    needs_manual_review = ambiguous_calc or low_confidence

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
    output_extractor: Optional[CalcExtractor] = None,
    golden_path: Path = GOLDEN_PATH,
) -> Dict[str, Any]:
    """Run the full evaluation and return a machine-readable payload."""
    settings = settings or get_settings()
    retriever = retriever or InsuranceRetriever(settings=settings)
    if fact_judge is None:
        fact_judge = SemanticFactJudge(settings=settings)
    if output_extractor is None:
        output_extractor = CalculationOutputExtractor(settings=settings)
    records = load_golden(golden_path)

    results = [
        evaluate_question(
            r, retriever, settings, llm=llm,
            fact_judge=fact_judge, output_extractor=output_extractor,
        )
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
