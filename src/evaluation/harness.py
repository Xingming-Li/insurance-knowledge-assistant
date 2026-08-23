"""Evaluation harness for the Insurance Knowledge Assistant.

Runs every question in ``eval/golden_qa.jsonl`` through the current retrieval
and generation layers and records what happened. Deterministic by design:
temperature 0, fixed model config, fixed retrieval-k.

IMPORTANT: the golden dataset is evaluation-only. The ``expected_answer`` is
NEVER shown to the retriever or the generation model — only ``question`` is.

Automatic (non-LLM) checks:
  * source recall  — how many required (document, section) pairs were retrieved
                     (pair-matched, so a same-named section in another document
                     does NOT count as a hit).
  * answer correctness — deterministic numeric/unit matching of ``key_facts``
                     against the generated answer. Non-numeric facts are not
                     auto-checkable and are surfaced for manual review.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from config import PROJECT_ROOT, Settings, get_settings
from generation.answer import generate_answer
from generation.prompts import INSUFFICIENT_EVIDENCE_MESSAGE
from retrieval.retriever import InsuranceRetriever

GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden_qa.jsonl"
DOCS_DIR = PROJECT_ROOT / "data" / "insurance_docs"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
DEFAULT_RESULTS_PATH = RESULTS_DIR / "latest.json"

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
# Source recall (pair-matched)
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
        "docs_retrieved": len(docs_hit),
        "docs_required": len(required_docs),
        "docs_recall": f"{len(docs_hit)}/{len(required_docs)}",
        "pairs_retrieved": len(pairs_hit),
        "pairs_required": len(required_pairs),
        "pairs_recall": f"{len(pairs_hit)}/{len(required_pairs)}",
        "complete": pairs_hit == required_pairs,
        "missing_pairs": [
            {"document_id": d, "section": s} for d, s in sorted(required_pairs - pairs_hit)
        ],
    }


# --------------------------------------------------------------------------
# Answer correctness (deterministic numeric/unit matching — no LLM judge)
# --------------------------------------------------------------------------

_NBSP = " "
_THIN = " "


def canonical_facts(text: str) -> Set[str]:
    """Extract normalized numeric/unit tokens from Swedish insurance text.

    Captures the facts that actually matter for these answers — amounts,
    percentages and durations — normalized so that '90 000 SEK', '90 000 kr'
    and '90000 SEK' all collapse to the same token. Deterministic and reliable;
    it deliberately ignores prose (which is left to manual review).
    """
    if not text:
        return set()
    t = text.replace(_NBSP, " ").replace(_THIN, " ").lower()
    tokens: Set[str] = set()

    # Percentages: "15 %" / "15%"
    for m in re.finditer(r"(\d+)\s*%", t):
        tokens.add(f"{int(m.group(1))}%")

    # Money: "90 000 SEK" / "1 500 kr"
    for m in re.finditer(r"(\d[\d ]*\d|\d)\s*(sek|kr)\b", t):
        tokens.add(f"{int(m.group(1).replace(' ', ''))}sek")

    # Durations: dagar / månader / veckor / år (with common Swedish suffixes)
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


def answer_correctness(
    record: Dict[str, Any], answer: str
) -> Optional[Dict[str, Any]]:
    """Deterministically check key_facts' numeric content against the answer.

    Returns None for abstention questions (key_facts there describe *why* the
    corpus is silent, so answer correctness is not meaningful).
    """
    if record.get("expected_behavior") != "answer":
        return None

    key_facts = record.get("key_facts", []) or []
    answer_tokens = canonical_facts(answer)

    supported: List[str] = []
    unsupported: List[str] = []
    uncheckable: List[str] = []

    for fact in key_facts:
        fact_tokens = canonical_facts(fact)
        if not fact_tokens:
            uncheckable.append(fact)  # no numeric content -> needs a human
        elif fact_tokens <= answer_tokens:
            supported.append(fact)
        else:
            unsupported.append(fact)

    checkable = len(supported) + len(unsupported)
    return {
        "key_facts_total": len(key_facts),
        "key_facts_checkable": checkable,
        "key_facts_supported": len(supported),
        "facts_recall": f"{len(supported)}/{checkable}" if checkable else "0/0",
        "unsupported_facts": unsupported,
        "uncheckable_facts": uncheckable,
    }


# --------------------------------------------------------------------------
# Per-question evaluation
# --------------------------------------------------------------------------

def evaluate_question(
    record: Dict[str, Any],
    retriever: Any,
    settings: Settings,
    llm: Any = None,
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

    recall = source_recall(record, retrieved_pairs)
    correctness = answer_correctness(record, result.answer)
    is_numeric = "calculation" in record
    behavior_matches = actual_behavior == record["expected_behavior"]

    needs_manual_review = not behavior_matches
    if record["expected_behavior"] == "answer":
        needs_manual_review = (
            needs_manual_review
            or is_numeric
            or (recall is not None and not recall["complete"])
            or (correctness is not None and correctness["unsupported_facts"] != [])
            or (correctness is not None and correctness["uncheckable_facts"] != [])
        )

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
            "source_recall": recall,
            "answer_correctness": correctness,
            "abstained": abstained,
            "behavior_matches_expected": behavior_matches,
            "is_numeric": is_numeric,
            "needs_manual_review": needs_manual_review,
        },
    }


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    answerable = [r for r in results if r["expected_behavior"] == "answer"]
    abstain = [r for r in results if r["expected_behavior"] == "abstain"]

    recalls = [r["checks"]["source_recall"] for r in answerable if r["checks"]["source_recall"]]
    complete = sum(1 for rc in recalls if rc["complete"])
    pairs_hit = sum(rc["pairs_retrieved"] for rc in recalls)
    pairs_req = sum(rc["pairs_required"] for rc in recalls)

    corrs = [r["checks"]["answer_correctness"] for r in answerable if r["checks"]["answer_correctness"]]
    facts_sup = sum(c["key_facts_supported"] for c in corrs)
    facts_chk = sum(c["key_facts_checkable"] for c in corrs)

    return {
        "n_questions": len(results),
        "behavior_match": sum(
            1 for r in results if r["checks"]["behavior_matches_expected"]
        ),
        "answerable": {
            "count": len(answerable),
            "answered": sum(1 for r in answerable if r["actual_behavior"] == "answer"),
            "source_complete": f"{complete}/{len(recalls)}",
            "pair_recall": f"{pairs_hit}/{pairs_req}",
            "key_fact_recall": f"{facts_sup}/{facts_chk}",
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
    golden_path: Path = GOLDEN_PATH,
) -> Dict[str, Any]:
    """Run the full evaluation and return a machine-readable payload."""
    settings = settings or get_settings()
    retriever = retriever or InsuranceRetriever(settings=settings)
    records = load_golden(golden_path)

    results = [evaluate_question(r, retriever, settings, llm=llm) for r in records]

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "temperature": settings.temperature,
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
    """Human-readable console report."""
    meta = payload["meta"]
    summary = payload["summary"]

    print("NordicPaws — Evaluation report")
    print(
        f"model={meta['chat_model']}  emb={meta['embedding_model']}  "
        f"temp={meta['temperature']}  k={meta['retrieval_k']}"
    )
    print("=" * 104)
    header = (
        f"{'QID':<4} {'type':<22} {'exp':<8} {'act':<8} "
        f"{'docs':<6} {'pairs':<7} {'facts':<7} {'review':<6}"
    )
    print(header)
    print("-" * 104)
    for r in payload["results"]:
        c = r["checks"]
        rc = c["source_recall"]
        co = c["answer_correctness"]
        docs = rc["docs_recall"] if rc else "-"
        pairs = rc["pairs_recall"] if rc else "-"
        facts = co["facts_recall"] if co else "-"
        print(
            f"{r['id']:<4} {str(r['type']):<22} "
            f"{r['expected_behavior']:<8} {r['actual_behavior']:<8} "
            f"{docs:<6} {pairs:<7} {facts:<7} "
            f"{('YES' if c['needs_manual_review'] else '-'):<6}"
        )

    print("=" * 104)
    a = summary["answerable"]
    ab = summary["abstain"]
    print(
        f"behavior match: {summary['behavior_match']}/{summary['n_questions']}   "
        f"answered: {a['answered']}/{a['count']}   "
        f"source complete: {a['source_complete']}   "
        f"pair recall: {a['pair_recall']}   "
        f"key-fact recall: {a['key_fact_recall']}   "
        f"abstain correct: {ab['correctly_abstained']}/{ab['count']}"
    )
    print(f"needs manual review: {', '.join(summary['needs_manual_review']) or 'none'}")
