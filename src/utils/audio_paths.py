from pathlib import Path
from typing import List, Optional


def _as_path(path_str: Optional[str]) -> Optional[Path]:
    if not isinstance(path_str, str) or not path_str.strip():
        return None
    return Path(path_str).expanduser()


def _dedupe_paths(candidates: List[Path]) -> List[Path]:
    deduped: List[Path] = []
    seen = set()

    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(resolved)

    return deduped


def _drop_wavs_segment(path: Path) -> Optional[Path]:
    """Also try the path with a `wavs/` segment removed.

    A common layout difference between how a dataset is packaged and how it is
    unpacked; harmless when it does not apply, since the original path is
    always tried first."""
    if path.parent.name == "wavs":
        return path.parent.parent / path.name

    if "wavs" not in path.parts:
        return None

    return Path(*[part for part in path.parts if part != "wavs"])


def candidate_audio_paths(
    audio_path: Optional[str],
    metadata_path: Optional[str] = None,
    source_file: Optional[str] = None,
) -> List[Path]:
    audio_ref = _as_path(audio_path)
    metadata_ref = _as_path(metadata_path)
    source_ref = _as_path(source_file)
    audio_name = audio_ref.name if audio_ref is not None else None

    if audio_name is None and source_ref is not None:
        audio_name = f"{source_ref.stem}.wav"

    candidates: List[Path] = []

    if audio_ref is not None:
        candidates.append(audio_ref)
        no_wavs_candidate = _drop_wavs_segment(audio_ref)
        if no_wavs_candidate is not None:
            candidates.append(no_wavs_candidate)

    if metadata_ref is not None and metadata_ref.parent.name == "metadata":
        if audio_name is not None:
            candidates.append(metadata_ref.parent.parent / audio_name)
        if source_ref is not None:
            candidates.append(metadata_ref.parent.parent / f"{source_ref.stem}.wav")

    return _dedupe_paths(candidates)


def resolve_audio_path(
    audio_path: Optional[str],
    metadata_path: Optional[str] = None,
    source_file: Optional[str] = None,
) -> Optional[Path]:
    candidates = candidate_audio_paths(audio_path, metadata_path, source_file)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if candidates:
        return candidates[0]

    return None


def normalize_optional_path(path_str: Optional[str]) -> Optional[str]:
    path = _as_path(path_str)
    if path is None:
        return None
    return str(path.resolve())
