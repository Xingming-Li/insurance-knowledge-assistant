"""Retrieval-only diagnostics for selected questions.

Deep-dive on why certain expected (document_id, section) pairs are or are not
retrieved. This script ONLY reads the vector store — it never calls the
generation model, the semantic fact judge, or evaluates answer correctness, and
it does not modify RETRIEVAL_K, the golden set, or the index.

It fetches the top-N (default 20) chunks per question purely as a diagnostic.

Run:
    PYTHONPATH=src python scripts/inspect_retrieval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Put ``src`` on the path when run as a plain script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import config  # noqa: E402
from retrieval.retriever import InsuranceRetriever  # noqa: E402

GOLDEN = ROOT / "eval" / "golden_qa.jsonl"
TARGET_QIDS = ["Q6", "Q7", "Q8", "Q10"]
TOP_N = 20


def load_golden():
    records = {}
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            records[r["id"]] = r
    return records


def preview(text: str, n: int = 50) -> str:
    return " ".join(text.split())[:n]


def collection_metric(vs) -> str:
    try:
        return str((vs._collection.metadata or {}).get("hnsw:space", "default (l2)"))
    except Exception:
        return "unknown"


def main() -> None:
    settings = config.get_settings()
    retriever = InsuranceRetriever(settings=settings)
    vs = retriever._get_vector_store()  # read-only

    records = load_golden()
    print(f"RETRIEVAL-ONLY diagnostics   top-N = {TOP_N}   "
          f"configured RETRIEVAL_K = {settings.retrieval_k} (unchanged)")
    print(f"Chroma distance metric: {collection_metric(vs)}   "
          "(relevance score: higher = more relevant)")

    for qid in TARGET_QIDS:
        rec = records[qid]
        question = rec["question"]
        expected_pairs = [
            (s["document_id"], s["section"]) for s in rec.get("expected_sources", [])
        ]

        results = vs.similarity_search_with_relevance_scores(question, k=TOP_N)
        # rank lookup for each (doc_id, section)
        ranked = []
        for rank, (doc, score) in enumerate(results, 1):
            m = doc.metadata or {}
            ranked.append((rank, score, m.get("document_id"), m.get("section"),
                           doc.page_content))

        print("\n" + "=" * 100)
        print(f"{qid}  [{rec['type']}]  k_configured={settings.retrieval_k}")
        print(f"Q: {question}")
        print("-" * 100)
        print(f"{'rank':<5}{'score':>7}  {'match':<6} {'document_id':<18} "
              f"{'section':<34} preview")
        print("-" * 100)
        for rank, score, doc_id, section, content in ranked:
            match = "  ***" if (doc_id, section) in expected_pairs else ""
            print(f"{rank:<5}{score:>7.3f}  {match:<6} {str(doc_id):<18} "
                  f"{str(section):<34.34} {preview(content)}")

        # ---- expected-pair report ----
        print("-" * 100)
        print("EXPECTED SOURCE PAIRS:")
        rank_by_pair = {(d, s): (rk, sc) for rk, sc, d, s, _ in ranked}
        found = 0
        found_within_k = 0
        for d, s in expected_pairs:
            hit = rank_by_pair.get((d, s))
            if hit:
                found += 1
                if hit[0] <= settings.retrieval_k:
                    found_within_k += 1
                print(f"  FOUND  rank {hit[0]:<3} score {hit[1]:.3f}   {d} | {s}")
            else:
                print(f"  NOT FOUND (absent from top {TOP_N})   {d} | {s}")

        total = len(expected_pairs)
        if found == 0:
            verdict = "expected evidence NOT retrieved at all (top-20)"
        elif found < total:
            verdict = "expected evidence PARTIALLY retrieved; some pairs missing from top-20"
        elif found_within_k < total:
            verdict = "expected evidence present but RANKED TOO LOW (outside k)"
        else:
            verdict = "expected evidence present within k"
        print(f"SUMMARY: {found}/{total} pairs in top-{TOP_N}, "
              f"{found_within_k}/{total} within k={settings.retrieval_k} -> {verdict}")


if __name__ == "__main__":
    main()
