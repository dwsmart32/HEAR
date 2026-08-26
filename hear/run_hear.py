#!/usr/bin/env python3
"""End-to-end HEAR evaluation: download from the Hub, run a model, score it.

    python examples/run_hear.py --model qwen2_5_omni-7b

Downloads PleasedPenguin/HEAR, writes the harness input file, runs inference with
whatever backend the registry says the model needs, and prints the paper's metrics.
Nothing here is specific to our machine -- every model id is a public Hub id.
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def build_input(out_path, split="test", limit=0):
    """Turn the Hub dataset into the jsonl the harness reads."""
    from datasets import load_dataset

    try:
        ds = load_dataset("PleasedPenguin/HEAR", split=split)
    except Exception as e:
        sys.exit(
            f"could not load PleasedPenguin/HEAR: {e}\n\n"
            "HEAR is a gated dataset. Two one-time steps:\n"
            "  1. open https://huggingface.co/datasets/PleasedPenguin/HEAR and\n"
            "     accept the data use agreement (approval is automatic)\n"
            "  2. hf auth login"
        )
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    audio_dir = os.path.join(os.path.dirname(out_path), "audio")
    os.makedirs(audio_dir, exist_ok=True)

    import soundfile as sf
    n = 0
    with open(out_path, "w") as w:
        for r in ds:
            wav = os.path.join(audio_dir, f"{r['question_id']}.wav")
            if not os.path.exists(wav):
                sf.write(wav, r["audio"]["array"], r["audio"]["sampling_rate"])
            w.write(json.dumps({
                "id":           r["question_id"],
                "clip_id":      r["clip_id"],
                "dataset":      r["source_corpus"],
                "axis":         r["taxonomy"],
                "subdimension": r["subtaxonomy"],
                "category":     rebuild_category(r),
                "options":      r["options"],
                "ground_truth": chr(65 + r["answer_idx"]),
                "instruction":  build_instruction(r),
                "audio_path":   os.path.abspath(wav),
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"[input] {n} rows -> {out_path}")
    return out_path


def rebuild_category(r):
    """Reassemble the original leaf label, e.g. "QR1_nonoverlap", "VCD_change", "VC".

    `task_name` already carries the sub-dimension prefix for QR and TR
    ("QR1", "TR1_first"), so it replaces the head instead of being appended.
    """
    sub, task, ov = r["subtaxonomy"], r["task_name"], r["overlap"]
    if sub == "VCD":
        return f"VCD_{task}"          # the task IS the regime; no overlap suffix
    if task == "none":
        return f"{sub}_{ov}" if ov else sub
    head = task if task.startswith(sub) else f"{sub}_{task}"
    return f"{head}_{ov}" if ov else head


def build_instruction(r):
    """Reconstruct the prompt text. `audio` already contains everything the
    model must hear, so the text only has to carry the question and options."""
    sub = r["subtaxonomy"]
    if sub == "CVA":
        head = ("Listen to the main audio, followed by the voice options audio where each "
                "option (A through E) is announced before its voice sample, and answer the "
                "following multiple-choice question.")
        body = f"\n\n{r['question']}\n"
    else:
        if sub in ("VL", "VCA"):
            head = ("Listen to the main audio, followed by a reference voice of the target "
                    "speaker, and answer the following multiple-choice question.")
        else:
            head = "Listen to the main audio and answer the following multiple-choice question."
        opts = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(r["options"]))
        body = f"\n\n{r['question']}\n\nOptions:\n{opts}\n"
    return head + body + '\nRespond in JSON format, e.g. {"Answer": "A"}'


def preflight(model, registry):
    """Fail before the multi-GB download if the backend is not installed."""
    import yaml
    reg = yaml.safe_load(open(registry))
    if model not in reg:
        sys.exit(f"{model!r} is not in {registry}. "
                 f"Available: {', '.join(sorted(reg))}")
    backend = reg[model].get("type")
    needed = {"vllm": "vllm", "huggingface": "transformers"}.get(backend)
    if needed:
        import importlib.util
        if importlib.util.find_spec(needed) is None:
            sys.exit(f"{model!r} uses the {backend} backend, which needs "
                     f"{needed!r}. requirements.txt installs neither vLLM nor "
                     f"transformers by default: pip install {needed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a key from registry.yaml")
    ap.add_argument("--workdir", default="runs/hear")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N rows")
    ap.add_argument("--skip-inference", action="store_true", help="score an existing run")
    a = ap.parse_args()

    if not a.skip_inference:
        preflight(a.model, os.path.join(ROOT, "registry.yaml"))

    os.makedirs(a.workdir, exist_ok=True)
    # The limit is part of the identity of the input file: reusing a 20-row
    # smoke-test file for a full run would silently score 20 items and print
    # it as a HEAR result.
    stem = "hear_input" + (f"_limit{a.limit}" if a.limit else "")
    inp = os.path.join(a.workdir, f"{stem}.jsonl")
    if not os.path.exists(inp):
        build_input(inp, limit=a.limit)

    results_dir = os.path.join(a.workdir, "results")
    if not a.skip_inference:
        subprocess.run([sys.executable, os.path.join(ROOT, "inference.py"),
                        "--model", a.model,
                        "--registry", os.path.join(ROOT, "registry.yaml"),
                        "--input", inp,
                        "--output_dir", results_dir], check=True)

    res = os.path.join(results_dir, f"{a.model}__{stem}_results.json")
    if not os.path.exists(res):
        if not os.path.isdir(results_dir):
            sys.exit(f"no results directory at {results_dir}; "
                     f"run without --skip-inference first")
        cand = [f for f in os.listdir(results_dir) if f.startswith(a.model)]
        if not cand:
            sys.exit(f"no results for {a.model} in {results_dir}")
        res = os.path.join(results_dir, cand[0])

    # eval_hear.py reads the results file alone: inference.py copies the input
    # fields (ground_truth, axis, subdimension, category) into each record.
    subprocess.run([sys.executable, os.path.join(HERE, "eval_hear.py"),
                    "--results", res,
                    "--out", os.path.join(a.workdir, f"{a.model}__summary.json")],
                   check=True)


if __name__ == "__main__":
    main()
