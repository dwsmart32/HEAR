import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.core.runner import ModelRunner
from src.utils.audio_paths import resolve_audio_path
from src.utils.utils import sanitize_filename


def resolve_project_path(path_str: str) -> Path:
    """Resolve a CLI path: absolute wins, then the caller's cwd, then the repo.

    Relative paths are the common case when pointing the harness at your own
    task, so cwd is tried before the repo root; the repo root remains a
    fallback so the bundled examples keep working from anywhere."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()

    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return from_cwd

    from_repo = (PROJECT_ROOT / path).resolve()
    if from_repo.exists():
        return from_repo

    # Neither exists yet (e.g. an output dir to be created): prefer cwd.
    return from_cwd


def resolve_data_path(path_str: str, input_path: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()

    for base_dir in (input_path.parent, PROJECT_ROOT, Path.cwd()):
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate

    return (PROJECT_ROOT / path).resolve()


def normalize_record_paths(item: dict, input_path: Path):
    if isinstance(item.get("source_file"), str):
        item["source_file"] = str(resolve_data_path(item["source_file"], input_path))

    if isinstance(item.get("metadata_path"), str):
        item["metadata_path"] = str(resolve_data_path(item["metadata_path"], input_path))

    if isinstance(item.get("audio_path"), str):
        original_audio_path = str(resolve_data_path(item["audio_path"], input_path))
        repaired_audio_path = resolve_audio_path(
            original_audio_path,
            item.get("metadata_path"),
            item.get("source_file"),
        )
        if repaired_audio_path is not None:
            item["audio_path"] = str(repaired_audio_path)
            return original_audio_path, str(repaired_audio_path)
        item["audio_path"] = original_audio_path
        return original_audio_path, original_audio_path

    return None, None


def main():
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model ID defined in registry")
    parser.add_argument("--registry", type=str, default="registry.yaml")
    parser.add_argument("--input", type=str, required=True,
                        help="JSONL task file: one query per line")
    parser.add_argument("--output_dir", type=str, default="inference_results/")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent API requests")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for local backends that support batch generation",
    )
    parser.add_argument(
        "--validate-input-only",
        action="store_true",
        help="Validate and repair dataset paths, then exit without loading a model.",
    )

    args = parser.parse_args()
    registry_path = resolve_project_path(args.registry)
    input_path = resolve_project_path(args.input)
    output_dir = resolve_project_path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    data = []
    repaired_audio_paths = set()
    missing_audio_paths = {}
    with input_path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                item['model_id'] = args.model
                original_audio_path, repaired_audio_path = normalize_record_paths(
                    item, input_path
                )
                if (
                    original_audio_path is not None
                    and repaired_audio_path is not None
                    and repaired_audio_path != original_audio_path
                ):
                    repaired_audio_paths.add((original_audio_path, repaired_audio_path))
                if isinstance(item.get("audio_path"), str) and not Path(
                    item["audio_path"]
                ).exists():
                    missing_audio_paths.setdefault(item["audio_path"], []).append(
                        item.get("id", "<unknown>")
                    )
                data.append(item)

    print(f"loaded {len(data)} samples from {input_path}")
    if repaired_audio_paths:
        print(
            f"Resolved {len(repaired_audio_paths)} audio paths "
            "relative to the input file."
        )
    if missing_audio_paths:
        examples = []
        for missing_path, sample_ids in list(missing_audio_paths.items())[:5]:
            examples.append(f"{missing_path} (sample_id={sample_ids[0]})")
        raise FileNotFoundError(
            "Input validation failed: "
            f"{len(missing_audio_paths)} unique audio files could not be resolved. "
            f"Examples: {'; '.join(examples)}"
        )
    if args.validate_input_only:
        print("Input validation completed successfully.")
        return

    runner = ModelRunner(
        model_name=args.model,
        registry_path=str(registry_path),
        concurrency=args.concurrency,
        batch_size=args.batch_size,
    )

    print(f"Initializing Runner: [{runner.__class__.__name__}] for model: {args.model}")

    results = runner.inference(data)

    # Post-process: split reasoning trace from answer for thinking models
    for item in results:
        resp = item.get("response", "")
        if isinstance(resp, str) and "</think>" in resp:
            parts = resp.split("</think>", 1)
            reasoning = parts[0].replace("<think>", "").strip()
            answer = parts[1].strip()
            item["reasoning_trace"] = reasoning
            item["response"] = answer

    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_filename(args.model)
    safe_input_name = sanitize_filename(input_path.stem)
    output_stem = f"{safe_name}__{safe_input_name}_results"
    output_json_path = output_dir / f"{output_stem}.json"

    with output_json_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} results to: {output_json_path}")

if __name__ == "__main__":
    main()
