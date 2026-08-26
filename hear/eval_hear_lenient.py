"""Lenient HEAR v2 eval: same logic as eval_hear.py + extra fallbacks
to recover parse fails from models that don't strictly follow format.

New rescue rules:
  - Strip ```json ... ``` / ``` ... ``` fenced blocks.
  - Strip escaped-quote wrappers ('{\\"Answer\\": ...}').
  - Parse Python-dict ({'Answer': '1'}) via ast.literal_eval.
  - Quoted digit value "1"..."5" → letter (1-indexed: 1=A, 2=B, ...).
  - Unquoted digit "Answer": 1 → also tries 1-indexed if 0-indexed gives '?'.
  - Prefer LAST standalone A-E letter (not first) since many models put it at end.

Outputs alongside the strict eval so you can compare rescue gains.
"""
import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_hear import _match_options, evaluate as strict_evaluate, LETTERS, extract_letter as extract_letter_strict

_RE_FENCE = re.compile(r"```(?:json|python)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_RE_JSON_ANS_LETTER = re.compile(
    r'["\']?\s*Answer\s*["\']?\s*:\s*["\']?\s*([A-E])\b',
    re.IGNORECASE,
)
_RE_JSON_DIGIT_QUOTED = re.compile(
    r'["\']?\s*Answer\s*["\']?\s*:\s*["\'\s]*([1-5])\b',
    re.IGNORECASE,
)
_RE_JSON_DIGIT_UNQUOTED = re.compile(
    r'["\']?\s*Answer\s*["\']?\s*:\s*([0-4])\b',
    re.IGNORECASE,
)
_RE_JSON_VALUE = re.compile(
    r'["\']?Answer["\']?\s*:\s*["\']?([^"\'}\n,]+?)["\']?\s*[,}\n]',
    re.IGNORECASE,
)
_RE_PAREN = re.compile(r'\(([A-E])\)')
_RE_STANDALONE = re.compile(r'\b([A-E])\b')


def _normalize(s):
    """Pre-process: strip fences, unescape quotes, collapse to a clean string."""
    if not s:
        return ""
    s = str(s).strip()
    # escape-unwrap: '{\"Answer\": \"2\"}' → '{"Answer": "2"}'
    if '\\"' in s:
        s = s.replace('\\"', '"')
    # try fence strip
    m = _RE_FENCE.search(s)
    if m:
        s = m.group(1).strip()
    return s


def _try_dict_parse(s):
    """Try ast.literal_eval on something that looks like a Python dict."""
    m = re.search(r'\{[^{}]*\}', s)
    if not m:
        return None
    blob = m.group(0)
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(blob)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if str(k).lower() == "answer":
                        return v
        except Exception:
            continue
    return None


def extract_letter_lenient(response, options=None) -> str:
    if not response:
        return ""
    # 0. Try strict eval first — never override a successful strict parse.
    strict = extract_letter_strict(response, options)
    if strict:
        return strict
    # If strict fails, run rescue on normalized text.
    s = _normalize(response)

    # 1. Direct JSON letter (in case normalization exposed it, e.g. after fence strip)
    m = _RE_JSON_ANS_LETTER.search(s)
    if m:
        return m.group(1).upper()

    # 2. Try dict parse (handles single-quoted, nested, etc.)
    v = _try_dict_parse(s)
    if v is not None:
        vs = str(v).strip()
        if len(vs) == 1 and vs.upper() in LETTERS:
            return vs.upper()
        # quoted digit 1-5 → 1-indexed letter
        if vs.isdigit():
            n = int(vs)
            if 1 <= n <= 5:
                return LETTERS[n - 1]
            if 0 <= n <= 4:
                return LETTERS[n]
        # match value against options text
        if options:
            letter = _match_options(vs, options)
            if letter:
                return letter

    # 3. Quoted digit direct regex
    m = _RE_JSON_DIGIT_QUOTED.search(s)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 5:
            return LETTERS[n - 1]

    # 4. Unquoted digit 0-4 (0-indexed)
    m = _RE_JSON_DIGIT_UNQUOTED.search(s)
    if m:
        return LETTERS[int(m.group(1))]

    # 5. JSON value against options text
    if options:
        m = _RE_JSON_VALUE.search(s)
        if m:
            letter = _match_options(m.group(1), options)
            if letter:
                return letter
        letter = _match_options(s, options)
        if letter:
            return letter

    # 6. Parenthesized letter
    m = _RE_PAREN.search(s)
    if m:
        return m.group(1).upper()

    # 7. Last standalone A-E letter (prefer end of response)
    matches = list(_RE_STANDALONE.finditer(s))
    if matches:
        return matches[-1].group(1).upper()

    return ""


def evaluate(results):
    by_cat = defaultdict(lambda: [0, 0])
    by_subdim = defaultdict(lambda: [0, 0])
    by_axis = defaultdict(lambda: [0, 0])
    by_type = defaultdict(lambda: [0, 0])
    overall = [0, 0, 0]
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
            pred = extract_letter_lenient(resp, item.get("options"))
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
                "response": str(resp)[:200],
            })

    return {
        "overall": {
            "accuracy": overall[0] / overall[1] if overall[1] else 0.0,
            "correct": overall[0],
            "total": overall[1],
            "parsed": overall[2],
            "parse_rate": overall[2] / overall[1] if overall[1] else 0.0,
        },
        "by_axis": {k: {"acc": v[0] / v[1] if v[1] else 0.0, "correct": v[0], "total": v[1]} for k, v in sorted(by_axis.items())},
        "by_subdim": {k: {"acc": v[0] / v[1] if v[1] else 0.0, "correct": v[0], "total": v[1]} for k, v in sorted(by_subdim.items())},
        "by_category": {k: {"acc": v[0] / v[1] if v[1] else 0.0, "correct": v[0], "total": v[1]} for k, v in sorted(by_cat.items())},
        "by_type": {k: {"acc": v[0] / v[1] if v[1] else 0.0, "correct": v[0], "total": v[1]} for k, v in sorted(by_type.items())},
        "parse_failures_sample": parse_failures[:20],
        "parse_failures_count": len(parse_failures),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    p = Path(args.results)
    data = json.load(open(p))
    summary = evaluate(data)
    out = Path(args.out) if args.out else p.with_suffix(".lenient.json")
    json.dump(summary, open(out, "w"), ensure_ascii=False, indent=2)
    o = summary["overall"]
    print(f'{p.name}: acc={o["accuracy"]*100:.2f}% parsed={o["parsed"]}/{o["total"]} pf={summary["parse_failures_count"]}  → {out.name}')


if __name__ == "__main__":
    main()
