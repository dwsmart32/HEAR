#!/usr/bin/env python3
"""Build a task JSONL from a folder of audio, then sweep models over it.

The point of this file is that it is boring: the harness has no idea what your
task is, so preparing one is just writing the JSONL below.

    python examples/run_custom_task.py --audio-dir ./clips \
        --instruction "How many distinct speakers are there?" \
        --models qwen2_5_omni-7b voxtral-mini-3b

With --ground-truth-csv (two columns: filename,answer) it also scores each run
with a plain accuracy, so you can see the whole loop end to end.
"""
import argparse
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


def build_jsonl(audio_dir, instruction, gt_map, out_path):
    clips = sorted(f for f in os.listdir(audio_dir) if f.lower().endswith(AUDIO_EXTS))
    if not clips:
        sys.exit(f"no audio files in {audio_dir} (looked for {', '.join(AUDIO_EXTS)})")

    with open(out_path, "w", encoding="utf-8") as w:
        for name in clips:
            rec = {
                "id": os.path.splitext(name)[0],
                "instruction": instruction,
                "audio_path": os.path.abspath(os.path.join(audio_dir, name)),
            }
            if name in gt_map:
                rec["ground_truth"] = gt_map[name]
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[task] {len(clips)} clips -> {out_path}")
    return out_path, clips


def score(results_path):
    """Plain accuracy over whatever `ground_truth` is present."""
    sys.path.insert(0, os.path.join(ROOT, "hear"))
    from eval_hear import extract_letter

    rows = json.load(open(results_path, encoding="utf-8"))
    scored = [r for r in rows if r.get("ground_truth")]
    if not scored:
        print("  (no ground_truth in this task, skipping scoring)")
        return
    ok = sum(1 for r in scored
             if extract_letter(str(r.get("response", "")), r.get("options"))
             == r["ground_truth"])
    print(f"  accuracy {100 * ok / len(scored):.1f}%  ({ok}/{len(scored)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--instruction", required=True,
                    help="the same prompt for every clip")
    ap.add_argument("--models", nargs="+", required=True,
                    help="one or more keys from registry.yaml")
    ap.add_argument("--ground-truth-csv", default=None,
                    help="two columns: filename,answer")
    ap.add_argument("--workdir", default="runs/custom")
    a = ap.parse_args()

    gt_map = {}
    if a.ground_truth_csv:
        with open(a.ground_truth_csv, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    gt_map[row[0].strip()] = row[1].strip()

    os.makedirs(a.workdir, exist_ok=True)
    task = os.path.join(a.workdir, "task.jsonl")
    build_jsonl(a.audio_dir, a.instruction, gt_map, task)

    results_dir = os.path.join(a.workdir, "results")
    for model in a.models:
        print(f"\n=== {model} ===")
        subprocess.run([sys.executable, os.path.join(ROOT, "inference.py"),
                        "--model", model,
                        "--registry", os.path.join(ROOT, "registry.yaml"),
                        "--input", task,
                        "--output_dir", results_dir], check=True)
        score(os.path.join(results_dir, f"{model}__task_results.json"))


if __name__ == "__main__":
    main()
