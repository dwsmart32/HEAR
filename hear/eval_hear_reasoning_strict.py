"""HEAR v2 Reasoning with original/hallucinated strict-pair evaluation.

For each model's result file, reports:
  - reasoning_original_acc: accuracy on '_original' items
  - reasoning_hallucinated_acc: accuracy on '_hallucinated' items
  - reasoning_strict_pair_acc: pair is correct iff BOTH original AND hallucinated
    members are correct (measures whether the model truly understands the
    semantics, not just guessing/patterns).

Usage:
  python eval_hear_reasoning_strict.py \
      --input runs/hear/hear_input.jsonl \
      --results runs/hear/results/<model>__hear_input_results.json
  # or to batch across all models:
  python eval_hear_reasoning_strict.py --input runs/hear/hear_input.jsonl \
      --results-glob 'runs/hear/results/*_results.json' --table
"""
import argparse
import glob
import json
import os
from pathlib import Path
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_hear import extract_letter


def is_original(i: str) -> bool:
    return i.endswith("_original")


def is_halluc(i: str) -> bool:
    return i.endswith("_hallucinated")


def pair_base(i: str):
    if is_original(i):
        return i[: -len("_original")], "original"
    if is_halluc(i):
        return i[: -len("_hallucinated")], "hallucinated"
    return None, None


def _resp_to_str(resp):
    if resp is None:
        return ""
    if isinstance(resp, dict):
        for k in ("prediction", "Answer", "answer"):
            v = resp.get(k)
            if isinstance(v, str) and v.strip():
                return f'{{"Answer": "{v.strip()}"}}'
        return json.dumps(resp, ensure_ascii=False)
    return resp if isinstance(resp, str) else str(resp)


def score(result_path: Path, reasoning_ids: set, input_by_id: dict):
    data = json.load(result_path.open())
    # Build id -> predicted letter
    pred_by_id = {}
    for r in data:
        rid = r["id"]
        if rid not in reasoning_ids:
            continue
        if "error" in r:
            pred_by_id[rid] = ""
            continue
        resp = _resp_to_str(r.get("response", ""))
        opts = input_by_id[rid].get("options")
        pred_by_id[rid] = extract_letter(resp, opts)

    # Pair up
    pairs = defaultdict(dict)
    for rid in reasoning_ids:
        base, tag = pair_base(rid)
        if base is None:
            continue
        pairs[base][tag] = rid

    orig_total = orig_correct = 0
    hal_total = hal_correct = 0
    pair_total = pair_correct = 0
    same_pred_pairs = 0           # model predicted identical letter on both sides
    same_pred_orig_correct = 0    # among same-pred, was original side correct
    same_pred_hal_correct = 0     # among same-pred, was hallucinated side correct
    diff_pred_pairs = 0
    diff_pred_both_correct = 0    # changed prediction AND both sides right (= strict)
    by_sub_pair = defaultdict(lambda: [0, 0])  # [correct, total]

    for base, pm in pairs.items():
        o_id = pm.get("original")
        h_id = pm.get("hallucinated")
        o_ok = False
        h_ok = False
        if o_id:
            gt = input_by_id[o_id]["ground_truth"]
            o_ok = pred_by_id.get(o_id, "") == gt
            orig_total += 1
            orig_correct += int(o_ok)
        if h_id:
            gt = input_by_id[h_id]["ground_truth"]
            h_ok = pred_by_id.get(h_id, "") == gt
            hal_total += 1
            hal_correct += int(h_ok)
        if o_id and h_id:
            pair_total += 1
            strict_ok = o_ok and h_ok
            pair_correct += int(strict_ok)
            sub = input_by_id[o_id]["subdimension"]
            by_sub_pair[sub][1] += 1
            by_sub_pair[sub][0] += int(strict_ok)
            o_pred = pred_by_id.get(o_id, "")
            h_pred = pred_by_id.get(h_id, "")
            if o_pred and o_pred == h_pred:
                same_pred_pairs += 1
                same_pred_orig_correct += int(o_ok)
                same_pred_hal_correct += int(h_ok)
            else:
                diff_pred_pairs += 1
                diff_pred_both_correct += int(strict_ok)

    return {
        "orig_acc": orig_correct / orig_total if orig_total else 0.0,
        "orig_n": orig_total,
        "hal_acc": hal_correct / hal_total if hal_total else 0.0,
        "hal_n": hal_total,
        "strict_pair_acc": pair_correct / pair_total if pair_total else 0.0,
        "pair_n": pair_total,
        # Consistency metrics: model prediction stayed identical across pair
        "same_pred_rate": same_pred_pairs / pair_total if pair_total else 0.0,
        "same_pred_n": same_pred_pairs,
        # Of same-pred pairs, accuracy on each side (GT differs, so at most one can be right)
        "same_pred_orig_acc": same_pred_orig_correct / same_pred_pairs if same_pred_pairs else 0.0,
        "same_pred_hal_acc": same_pred_hal_correct / same_pred_pairs if same_pred_pairs else 0.0,
        "diff_pred_rate": diff_pred_pairs / pair_total if pair_total else 0.0,
        "diff_pred_both_correct_acc": diff_pred_both_correct / diff_pred_pairs if diff_pred_pairs else 0.0,
        "by_sub_pair": {k: {"acc": v[0]/v[1] if v[1] else 0.0, "n": v[1]}
                        for k, v in sorted(by_sub_pair.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="HEAR v2 input JSONL")
    ap.add_argument("--results", nargs="*", default=[])
    ap.add_argument("--results-glob", default=None)
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()

    # Load input
    input_by_id = {}
    with open(args.input) as f:
        for line in f:
            r = json.loads(line)
            input_by_id[r["id"]] = r
    reasoning_ids = {rid for rid, rec in input_by_id.items()
                     if rec.get("axis") == "Reasoning"}

    paths = list(args.results)
    if args.results_glob:
        paths.extend(glob.glob(args.results_glob))
    paths = sorted(set(paths))

    rows = []
    for p in paths:
        name = os.path.basename(p).replace("__hear_input_results.json", "")
        try:
            s = score(Path(p), reasoning_ids, input_by_id)
        except Exception as e:
            print(f"skip {name}: {e}")
            continue
        rows.append((name, s))

    if args.table:
        # Sort by strict pair acc desc
        rows.sort(key=lambda r: -r[1]["strict_pair_acc"])
        print(f'{"model":<30}  orig%  hall%  strict%  sameP%  sameP_orig%  sameP_hal%  diffP_strict%')
        for name, s in rows:
            print(f'{name:<30}  {s["orig_acc"]*100:5.1f}  {s["hal_acc"]*100:5.1f}   {s["strict_pair_acc"]*100:5.1f}   {s["same_pred_rate"]*100:5.1f}        {s["same_pred_orig_acc"]*100:5.1f}       {s["same_pred_hal_acc"]*100:5.1f}       {s["diff_pred_both_correct_acc"]*100:5.1f}')
        print()
        # Also per-subdimension strict table
        subs = set()
        for _, s in rows: subs.update(s["by_sub_pair"].keys())
        subs = sorted(subs)
        hdr = ["model"] + [f"{s}%" for s in subs]
        print("-- strict-pair by subdimension --")
        print("  ".join(f"{h:>10}" if i else f"{h:<30}" for i, h in enumerate(hdr)))
        for name, s in rows:
            line = [f"{name:<30}"]
            for sd in subs:
                v = s["by_sub_pair"].get(sd)
                line.append(f"{v['acc']*100:>9.1f}" if v else f"{'-':>9}")
            print("  ".join(line))
    else:
        for name, s in rows:
            print(name, json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
