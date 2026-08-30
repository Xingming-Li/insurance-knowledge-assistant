"""Retrieval-only comparison: baseline dense vs decomposed retrieval.

No answer generation, no semantic judging. Uses the LLM decomposer only (that
IS the retrieval strategy under test). Budget-matched: baseline top-N vs
decomposed final set (<= max_chunks).

Run:
    PYTHONPATH=src python scripts/compare_retrieval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import config
from retrieval.decomposed import DecomposedRetriever
from retrieval.retriever import InsuranceRetriever

GOLDEN = ROOT / "eval" / "golden_qa.jsonl"
BUDGET = 8
FOCUS = {"Q6", "Q7", "Q8", "Q10"}


def load_answerable():
    out = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            if r["expected_behavior"] == "answer":
                out.append(r)
    return out


def expected_pairs(rec):
    return {(s["document_id"], s["section"]) for s in rec.get("expected_sources", [])}


def main() -> None:
    settings = config.get_settings()
    records = load_answerable()

    baseline = InsuranceRetriever(settings=settings)
    decomposed = DecomposedRetriever(settings=settings, k_sub=3, max_chunks=BUDGET)

    strat = {
        "baseline (top-%d)" % BUDGET: {"pairs": 0, "complete": 0, "chunks": 0, "nonmatch": 0, "per_q": []},
        "decomposed (<=%d)" % BUDGET: {"pairs": 0, "complete": 0, "chunks": 0, "nonmatch": 0, "per_q": []},
    }
    total_pairs = sum(len(expected_pairs(r)) for r in records)

    focus_lines = []

    for rec in records:
        qid = rec["id"]
        exp = expected_pairs(rec)

        # --- baseline ---
        b_docs = baseline.retrieve(rec["question"], k=BUDGET)
        b_pairs = [((d.metadata or {}).get("document_id"), (d.metadata or {}).get("section")) for d in b_docs]
        b_hit = exp & set(b_pairs)

        # --- decomposed ---
        detailed = decomposed.retrieve_detailed(rec["question"])
        d_final = detailed["final"]
        d_pairs = [(h["document_id"], h["section"]) for h in d_final]
        d_hit = exp & set(d_pairs)

        for name, hitset, pairs in (
            (list(strat)[0], b_hit, b_pairs),
            (list(strat)[1], d_hit, d_pairs),
        ):
            s = strat[name]
            s["pairs"] += len(hitset)
            s["complete"] += 1 if hitset == exp else 0
            s["chunks"] += len(pairs)
            s["nonmatch"] += sum(1 for p in pairs if p not in exp)
            s["per_q"].append((qid, f"{len(hitset)}/{len(exp)}"))

        if qid in FOCUS:
            focus_lines.append((qid, rec, exp, b_docs, detailed))

    # ---- headline table ----
    print(f"Answerable questions: {len(records)}   total expected pairs: {total_pairs}   budget: {BUDGET}\n")
    print(f"{'strategy':<20} {'pairs/'+str(total_pairs):<10} {'complete/'+str(len(records)):<12} "
          f"{'chunks':<8} {'nonmatch':<9} {'noise_ratio':<11}")
    print("-" * 74)
    for name, s in strat.items():
        noise = s["nonmatch"] / s["chunks"] if s["chunks"] else 0
        print(f"{name:<20} {str(s['pairs']):<10} {str(s['complete']):<12} "
              f"{str(s['chunks']):<8} {str(s['nonmatch']):<9} {noise:<11.3f}")

    print("\nPER-QUESTION SOURCE-PAIR RECALL")
    print(f"{'QID':<5} {'baseline':<10} {'decomposed':<10}")
    b_per = dict(strat[list(strat)[0]]["per_q"])
    d_per = dict(strat[list(strat)[1]]["per_q"])
    for rec in records:
        q = rec["id"]
        print(f"{q:<5} {b_per[q]:<10} {d_per[q]:<10}")

    # ---- focus questions ----
    for qid, rec, exp, b_docs, detailed in focus_lines:
        print("\n" + "=" * 96)
        print(f"{qid}: {rec['question']}")
        print(f"expected pairs ({len(exp)}): " + "; ".join(f"{d} | {s}" for d, s in sorted(exp)))

        print("\n  DECOMPOSITION:")
        print(f"    species = {detailed['decomposition'].get('species')}")
        for i, sq in enumerate(detailed["subqueries"]):
            from retrieval.decomposed import build_filter
            flt = build_filter(detailed["decomposition"].get("species"), sq.get("scope"))
            print(f"    [{i}] ({sq['scope']}) filter={flt}")
            print(f"        query: {sq['query']}")

        print("\n  DECOMPOSED final set (rank score doc | section  <- subquery):")
        for rank, h in enumerate(detailed["final"], 1):
            mark = "  ***" if (h["document_id"], h["section"]) in exp else "     "
            print(f"    {rank:<2}{mark} {h['score']:.3f} {h['document_id']} | {h['section']}"
                  f"   <- sq[{h['subquery_idx']}]")

        d_pairs = {(h["document_id"], h["section"]) for h in detailed["final"]}
        print("\n  EXPECTED PAIR RETRIEVAL:")
        for d, s in sorted(exp):
            in_dec = "yes" if (d, s) in d_pairs else "NO"
            b_pairs = [((x.metadata or {}).get("document_id"), (x.metadata or {}).get("section")) for x in b_docs]
            in_base = "yes" if (d, s) in b_pairs else "NO"
            print(f"    {d} | {s}   baseline={in_base}  decomposed={in_dec}")

        irrelevant = [(h["document_id"], h["section"]) for h in detailed["final"]
                      if (h["document_id"], h["section"]) not in exp]
        print(f"\n  IRRELEVANT chunks in decomposed final ({len(irrelevant)}): "
              + ("; ".join(f"{d}|{s}" for d, s in irrelevant) or "none"))


if __name__ == "__main__":
    main()
