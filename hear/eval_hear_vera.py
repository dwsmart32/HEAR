#!/usr/bin/env python3
"""Ver-A: the headline HEAR metric, where a counterfactual pair counts once.

A model that reads the transcript and ignores the voices can get one member of
a hallucination pair right by luck. Ver-A refuses to reward that: a Reasoning
pair scores only if BOTH members are correct, and the pair is then one unit.

    units = non-Reasoning items + Reasoning pairs
          = 1,235 + 580 = 1,815     (on the full HEAR test split)

Usage:
    python eval_hear_vera.py --results runs/hear/results/<model>__*.json
    python eval_hear_vera.py --results ... --out vera.json

Pairs are matched on question_id: HEAR names the two members of a pair
`<base>_original` and `<base>_hallucinated`.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_hear import extract_letter

PAIR_SUFFIXES = ("_original", "_hallucinated")


def pair_base(qid: str):
    """Return the shared stem of a pair member, or None if not a pair member."""
    for suffix in PAIR_SUFFIXES:
        if qid.endswith(suffix):
            return qid[: -len(suffix)]
    return None


def is_correct(item):
    gt = item.get("ground_truth", "")
    if not gt:
        return None
    if "error" in item:
        return False
    resp = item.get("response", "") or ""
    if isinstance(resp, dict):
        val = resp.get("prediction") or resp.get("Answer") or resp.get("answer") or ""
        resp = f'{{"Answer": "{val}"}}' if val else json.dumps(resp, ensure_ascii=False)
    return extract_letter(resp, item.get("options")) == gt


def evaluate(results):
    singles, pairs = [], defaultdict(dict)
    skipped = 0

    for item in results:
        ok = is_correct(item)
        if ok is None:
            skipped += 1
            continue
        qid = str(item.get("id", ""))
        base = pair_base(qid)
        if item.get("axis") == "Reasoning" and base:
            pairs[base][qid[len(base):]] = ok
        else:
            singles.append(ok)

    complete = {b: v for b, v in pairs.items() if len(v) == 2}
    orphans = sorted(set(pairs) - set(complete))

    pair_ok = sum(1 for v in complete.values() if all(v.values()))
    units = len(singles) + len(complete)
    correct = sum(singles) + pair_ok

    return {
        "ver_a": 100.0 * correct / units if units else 0.0,
        "units": units,
        "correct": correct,
        "non_reasoning_items": len(singles),
        "non_reasoning_correct": sum(singles),
        "reasoning_pairs": len(complete),
        "reasoning_pairs_correct": pair_ok,
        "orphan_pairs": orphans,
        "skipped_no_ground_truth": skipped,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="inference results JSON")
    ap.add_argument("--out", default=None, help="optional summary JSON output")
    a = ap.parse_args()

    results = json.load(open(a.results, encoding="utf-8"))
    if not isinstance(results, list):
        results = results.get("results", results.get("records", []))
    s = evaluate(results)

    if s["skipped_no_ground_truth"] and not s["units"]:
        sys.exit(f"no scorable records in {a.results}: every item is missing "
                 f"`ground_truth`. Ver-A is defined for HEAR only.")

    print(f"Ver-A  {s['ver_a']:.1f}%   ({s['correct']}/{s['units']} units)")
    print(f"  non-Reasoning items  {s['non_reasoning_correct']:5d}/{s['non_reasoning_items']}")
    print(f"  Reasoning pairs      {s['reasoning_pairs_correct']:5d}/{s['reasoning_pairs']}"
          f"   (both members correct)")
    if s["orphan_pairs"]:
        print(f"  WARNING: {len(s['orphan_pairs'])} incomplete pairs excluded, "
              f"e.g. {s['orphan_pairs'][0]}")
    if s["skipped_no_ground_truth"]:
        print(f"  {s['skipped_no_ground_truth']} records skipped (no ground_truth)")

    if a.out:
        Path(a.out).write_text(json.dumps(s, indent=2), encoding="utf-8")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
