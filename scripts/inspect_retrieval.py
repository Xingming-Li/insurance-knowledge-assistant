"""Retrieval relevance diagnostics

Runs every question in ``eval/golden_qa.jsonl`` through the current
retriever and reports the vector-store relevance scores, so we can look
at the ACTUAL score distribution for answerable vs. insufficient-evidence
questions before choosing any abstention threshold.

This script only READS from the index; it doesn't change retrieval behaviour.

One measurement limitation: this script only reports document-level hits
(expected_doc_retrieved), not section-level.

Run:
    PYTHONPATH=src python scripts/inspect_retrieval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

# Put ``src`` on the path when run as a plain script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

sys.stdout.reconfigure(encoding="utf-8")

import config
from retrieval.retriever import InsuranceRetriever  

GOLDEN = ROOT / "eval" / "golden_qa.jsonl"


def load_golden():
    records = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def preview(text: str, n: int = 50) -> str:
    return " ".join(text.split())[:n]


def collection_metric(vector_store) -> str:
    try:
        meta = vector_store._collection.metadata or {}
        return str(meta.get("hnsw:space", "default (l2)"))
    except Exception:
        return "unknown"


def main() -> None:
    settings = config.get_settings()
    k = settings.retrieval_k

    retriever = InsuranceRetriever(settings=settings)
    vs = retriever._get_vector_store()  # read-only use of the underlying store

    records = load_golden()
    print(f"Questions in golden set: {len(records)}   |   top-k = {k}")
    print(f"Chroma distance metric: {collection_metric(vs)}")
    print("Note: relevance score is higher = more relevant (approx 0-1).\n")

    header = f"{'QID':<4} {'expected':<9} {'rank':<4} {'score':>7}  {'document_id':<18} {'section':<32} preview"
    print(header)
    print("-" * len(header))

    per_question = []  # (qid, expected, top_score, expected_hit)
    scores_by_group = {"answer": [], "abstain": []}

    for rec in records:
        qid = rec["id"]
        expected = rec["expected_behavior"]
        question = rec["question"]
        expected_doc_ids = {
            s.get("document_id") for s in rec.get("expected_sources", [])
        }

        results = vs.similarity_search_with_relevance_scores(question, k=k)

        top_score = results[0][1] if results else float("nan")
        scores_by_group[expected].append(top_score)

        hit = False
        for rank, (doc, score) in enumerate(results, 1):
            m = doc.metadata or {}
            doc_id = m.get("document_id", "?")
            section = (m.get("section") or "-")
            if doc_id in expected_doc_ids:
                hit = True
            print(
                f"{qid:<4} {expected:<9} {rank:<4} {score:>7.3f}  "
                f"{doc_id:<18} {section:<32.32} {preview(doc.page_content)}"
            )
        print()
        per_question.append((qid, expected, top_score, hit))

    # ---- Summary -------------------------------------------------------
    print("=" * 70)
    print("TOP-1 SCORE SUMMARY BY EXPECTED BEHAVIOR")
    for group in ("answer", "abstain"):
        vals = [v for v in scores_by_group[group] if v == v]  # drop NaN
        if vals:
            print(
                f"  {group:<8}  n={len(vals):<2}  "
                f"min={min(vals):.3f}  mean={mean(vals):.3f}  max={max(vals):.3f}"
            )

    print("\nEXPECTED-EVIDENCE RETRIEVAL (answerable questions only)")
    for qid, expected, top, hit in per_question:
        if expected == "answer":
            print(f"  {qid:<4} top1={top:.3f}  expected_doc_retrieved={hit}")

    print("\nABSTAIN QUESTIONS (should have NO strong match)")
    for qid, expected, top, hit in per_question:
        if expected == "abstain":
            print(f"  {qid:<4} top1={top:.3f}")


if __name__ == "__main__":
    main()
