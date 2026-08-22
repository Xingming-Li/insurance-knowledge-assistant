"""Evaluation harness for the Insurance Knowledge Assistant

Runs every question in ``eval/golden_qa.jsonl`` through the current
retrieval and generation layers and records what happened. Deterministic
by design: temperature 0, fixed model config, fixed retrieval-k.

IMPORTANT: the golden dataset is evaluation-only. The ``expected_answer``
is NEVER shown to the retriever or the generation model.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# Deterministic markers that indicate the assistant declares the evidence
# insufficient (either the canonical message or a model-generated equivalent).
_ABSTENTION_MARKERS = (
    "inte tillräcklig",
    "inte räcker",
    "otillräcklig",
    "kan inte besvara",
    "kan tyvärr inte besvara",
    "innehåller ingen information",
    "saknar information",
    "framgår inte av",
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


def _retrieval_checks(record: Dict[str, Any], retrieved_docs, retrieved_sections):
    """Document- and section-level retrieval checks.

    Returns (doc_hit, section_hit). Both None when the question has no
    expected sources (i.e. an abstention question).
    """
    expected_sources = record.get("expected_sources", []) or []
    if not expected_sources:
        return None, None
    exp_docs = {s.get("document_id") for s in expected_sources}
    exp_secs = {s.get("section") for s in expected_sources}
    doc_hit = any(d in exp_docs for d in retrieved_docs)
    section_hit = any(s in exp_secs for s in retrieved_sections)
    return doc_hit, section_hit


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

    abstained = detect_abstention(result.answer) or not result.used_evidence
    actual_behavior = "abstain" if abstained else "answer"

    doc_hit, section_hit = _retrieval_checks(
        record, retrieved_documents, retrieved_sections
    )
    is_numeric = "calculation" in record
    behavior_matches = actual_behavior == record["expected_behavior"]

    needs_manual_review = (
        is_numeric
        or not behavior_matches
        or doc_hit is False
        or section_hit is False
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
            "expected_source_doc_retrieved": doc_hit,
            "expected_section_retrieved": section_hit,
            "abstained": abstained,
            "behavior_matches_expected": behavior_matches,
            "is_numeric": is_numeric,
            "needs_manual_review": needs_manual_review,
        },
    }


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    def rate(subset, key):
        vals = [r["checks"][key] for r in subset if r["checks"][key] is not None]
        return round(sum(1 for v in vals if v) / len(vals), 3) if vals else None

    answerable = [r for r in results if r["expected_behavior"] == "answer"]
    abstain = [r for r in results if r["expected_behavior"] == "abstain"]

    return {
        "n_questions": len(results),
        "behavior_match": sum(
            1 for r in results if r["checks"]["behavior_matches_expected"]
        ),
        "answerable": {
            "count": len(answerable),
            "doc_hit_rate": rate(answerable, "expected_source_doc_retrieved"),
            "section_hit_rate": rate(answerable, "expected_section_retrieved"),
            "answered_rate": round(
                sum(1 for r in answerable if r["actual_behavior"] == "answer")
                / len(answerable),
                3,
            )
            if answerable
            else None,
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
    print("=" * 100)
    header = (
        f"{'QID':<4} {'type':<22} {'exp':<8} {'act':<8} "
        f"{'doc':<5} {'sec':<5} {'review':<6} answer preview"
    )
    print(header)
    print("-" * 100)
    for r in payload["results"]:
        c = r["checks"]

        def mark(v):
            return "-" if v is None else ("yes" if v else "NO")

        preview = " ".join((r["answer"] or "").split())[:44]
        print(
            f"{r['id']:<4} {str(r['type']):<22} "
            f"{r['expected_behavior']:<8} {r['actual_behavior']:<8} "
            f"{mark(c['expected_source_doc_retrieved']):<5} "
            f"{mark(c['expected_section_retrieved']):<5} "
            f"{('YES' if c['needs_manual_review'] else '-'):<6} {preview}"
        )

    print("=" * 100)
    a = summary["answerable"]
    ab = summary["abstain"]
    print(
        f"behavior match: {summary['behavior_match']}/{summary['n_questions']}   "
        f"answerable doc-hit={a['doc_hit_rate']} section-hit={a['section_hit_rate']} "
        f"answered={a['answered_rate']}   "
        f"abstain correct={ab['correctly_abstained']}/{ab['count']}"
    )
    print(f"needs manual review: {', '.join(summary['needs_manual_review']) or 'none'}")
