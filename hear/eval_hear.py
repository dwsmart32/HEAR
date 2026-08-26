"""Evaluate HEAR benchmark inference results.

Reads an inference result JSON (output of inference.py: list of records with
'response' and 'ground_truth' fields), parses the predicted letter (A-E),
and reports accuracy overall + per axis + per subdimension + per category.

Usage:
    python eval_hear.py --results inference_results/<model>__inference_results.json
    python eval_hear.py --results <path> --out <summary.json>
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

LETTERS = ("A", "B", "C", "D", "E")

# Strict JSON {"Answer": "A"} (allows quotes, whitespace, optional period)
_RE_JSON_ANS = re.compile(
    r'["\']?\s*Answer\s*["\']?\s*:\s*["\']?\s*([A-E])\b',
    re.IGNORECASE,
)
# Number-form answer: {"Answer": 0} → A
_RE_JSON_DIGIT = re.compile(
    r'["\']?\s*Answer\s*["\']?\s*:\s*([0-4])\b',
)
# Free-form "(A)" or "Answer is A" or just "A."
_RE_PAREN = re.compile(r'\(([A-E])\)')
_RE_STANDALONE = re.compile(r'\b([A-E])\b')
# Generic "Answer": <value> capture (value can be anything until quote/brace/comma/newline)
_RE_JSON_ANY_VALUE = re.compile(
    r'["\']?Answer["\']?\s*:\s*"?([^"}\n,]+?)"?\s*[,}\n]',
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    return str(s).strip().strip("\"'").lower()


def _norm_ws(s: str) -> str:
    """Normalize: lowercase, strip quotes, collapse all whitespace."""
    return re.sub(r"\s+", "", _norm(s))


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _numbers_in(s: str):
    """Extract numeric tokens as floats. '[14.5s - 19.0s]' → [14.5, 19.0]."""
    return [float(x) for x in _NUM_RE.findall(s)]


def _match_options(candidate: str, options) -> str:
    """Try to match a free-form candidate against the option texts.
    Returns letter on unique match, else ''.
    Strategies in order: whitespace-normalized exact / substring (both
    directions) / numeric-set equality (handles '[14.5s-19.0s]' vs
    '[14.5s - 19.0s]' and other formatting variants)."""
    if not candidate or not options:
        return ""
    cand_ws = _norm_ws(candidate)
    opts_ws = [_norm_ws(o) for o in options]

    # 1) exact (whitespace-normalized)
    for i, o in enumerate(opts_ws):
        if cand_ws and cand_ws == o:
            return LETTERS[i]
    # 2) option text appears inside candidate
    matches = [i for i, o in enumerate(opts_ws) if o and o in cand_ws]
    if len(matches) == 1:
        return LETTERS[matches[0]]
    # 3) candidate appears inside option
    matches = [i for i, o in enumerate(opts_ws) if cand_ws and cand_ws in o]
    if len(matches) == 1:
        return LETTERS[matches[0]]
    # 4) numeric-set equality (good for time-range options)
    cand_nums = _numbers_in(candidate)
    if len(cand_nums) >= 2:
        opts_nums = [_numbers_in(o) for o in options]
        cand_sorted = sorted(cand_nums)
        matches = [i for i, on in enumerate(opts_nums)
                   if on and sorted(on) == cand_sorted]
        if len(matches) == 1:
            return LETTERS[matches[0]]
    return ""


def extract_letter(response: str, options=None) -> str:
    """Return predicted letter A-E or '' if cannot parse.
    `options` (list of option texts) enables value→option-text matching."""
    if not response:
        return ""
    s = response.strip()

    # 1. Strict JSON-style {"Answer": "A"}
    m = _RE_JSON_ANS.search(s)
    if m:
        return m.group(1).upper()

    # 2. JSON with digit {"Answer": 0}
    m = _RE_JSON_DIGIT.search(s)
    if m:
        return LETTERS[int(m.group(1))]

    # 3. Loose JSON parse
    try:
        json_match = re.search(r'\{[^{}]*\}', s)
        if json_match:
            obj = json.loads(json_match.group(0))
            for k, v in obj.items():
                if k.lower() == "answer":
                    if isinstance(v, str) and v.strip().upper() in LETTERS:
                        return v.strip().upper()
                    if isinstance(v, int) and 0 <= v <= 4:
                        return LETTERS[v]
    except Exception:
        pass

    # 4. Parenthesized letter (B)
    m = _RE_PAREN.search(s)
    if m:
        return m.group(1).upper()

    # 5. JSON value matched against options text (Gemma "{"Answer":"3"}" → letter)
    if options:
        m = _RE_JSON_ANY_VALUE.search(s)
        if m:
            letter = _match_options(m.group(1), options)
            if letter:
                return letter
        # 6. Whole response matched against options (audio-flamingo style: outputs option text directly)
        letter = _match_options(s, options)
        if letter:
            return letter

    # 7. First standalone letter (last-resort fallback; risk of false positives)
    m = _RE_STANDALONE.search(s)
    if m:
        return m.group(1).upper()

    return ""


def evaluate(results):
    by_cat = defaultdict(lambda: [0, 0])      # [correct, total]
    by_subdim = defaultdict(lambda: [0, 0])
    by_axis = defaultdict(lambda: [0, 0])
    by_type = defaultdict(lambda: [0, 0])
    overall = [0, 0, 0]                        # correct, total, parsed
    parse_failures = []

    for item in results:
        gt = item.get("ground_truth", "")
        if not gt:
            continue
        resp = item.get("response", "") or ""
        if isinstance(resp, dict):
            pred_val = resp.get("prediction") or resp.get("Answer") or resp.get("answer") or ""
            resp = json.dumps(resp, ensure_ascii=False) if not pred_val else f'{{"Answer": "{pred_val}"}}'
        if "error" in item:
            pred = ""
        else:
            pred = extract_letter(resp, item.get("options"))
        is_correct = (pred == gt)

        overall[1] += 1
        if pred:
            overall[2] += 1
        if is_correct:
            overall[0] += 1

        cat = item.get("category", "?")
        subdim = item.get("subdimension", "?")
        axis = item.get("axis", "?")
        typ = item.get("type", "?")

        for bucket, key in (
            (by_cat, cat), (by_subdim, subdim),
            (by_axis, axis), (by_type, typ),
        ):
            bucket[key][1] += 1
            if is_correct:
                bucket[key][0] += 1

        if not pred:
            parse_failures.append({
                "id": item.get("id"),
                "category": cat,
                "response": resp[:200],
            })

    return {
        "overall": {
            "accuracy": overall[0] / overall[1] if overall[1] else 0.0,
            "correct": overall[0],
            "total": overall[1],
            "parsed": overall[2],
            "parse_rate": overall[2] / overall[1] if overall[1] else 0.0,
        },
        "by_axis": {k: {"acc": v[0] / v[1] if v[1] else 0.0,
                        "correct": v[0], "total": v[1]}
                    for k, v in sorted(by_axis.items())},
        "by_subdim": {k: {"acc": v[0] / v[1] if v[1] else 0.0,
                          "correct": v[0], "total": v[1]}
                      for k, v in sorted(by_subdim.items())},
        "by_category": {k: {"acc": v[0] / v[1] if v[1] else 0.0,
                            "correct": v[0], "total": v[1]}
                        for k, v in sorted(by_cat.items())},
        "by_type": {k: {"acc": v[0] / v[1] if v[1] else 0.0,
                        "correct": v[0], "total": v[1]}
                    for k, v in sorted(by_type.items())},
        "parse_failures_sample": parse_failures[:20],
        "parse_failures_count": len(parse_failures),
    }


def fmt_block(title, table):
    print(f"\n--- {title} ---")
    for k, v in table.items():
        print(f"  {k:30s}  {v['acc']*100:5.1f}%  ({v['correct']}/{v['total']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="inference results JSON")
    ap.add_argument("--out", default=None, help="optional summary JSON output")
    args = ap.parse_args()

    p = Path(args.results)
    if not p.exists():
        sys.exit(f"results not found: {p}")
    with p.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        sys.exit("expected a list of records")

    print(f"Loaded {len(data)} results from {p}")
    summary = evaluate(data)

    o = summary["overall"]
    print(f"\n=== Overall ===")
    print(f"  Accuracy: {o['accuracy']*100:.2f}%  ({o['correct']}/{o['total']})")
    print(f"  Parse rate: {o['parse_rate']*100:.2f}%  ({o['parsed']}/{o['total']})")

    fmt_block("By Axis", summary["by_axis"])
    fmt_block("By Subdimension", summary["by_subdim"])
    fmt_block("By Category", summary["by_category"])
    fmt_block("By Type (audio layout)", summary["by_type"])

    if summary["parse_failures_count"]:
        print(f"\nParse failures: {summary['parse_failures_count']}")

    out_path = Path(args.out) if args.out else p.with_suffix(".eval.json")
    with out_path.open("w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved summary → {out_path}")


if __name__ == "__main__":
    main()
