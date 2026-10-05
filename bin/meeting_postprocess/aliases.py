"""Load an explicit per-meeting speaker mapping, without guessing names."""
from __future__ import annotations

import json
from pathlib import Path

from .normalization import SPEAKER_LABEL, normalize_text


def load_aliases(transcript_dir: Path, explicit_path: Path | None = None) -> dict[str, str]:
    path = explicit_path if explicit_path is not None else transcript_dir / "speaker_aliases.json"
    if explicit_path is None and not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object mapping SPEAKER_XX labels to names")
    aliases = {}
    for label, name in data.items():
        if not SPEAKER_LABEL.fullmatch(label):
            raise ValueError(f"{path}: invalid speaker label {label!r}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}: {label} must map to a nonempty name")
        name = normalize_text(name.strip())
        if SPEAKER_LABEL.search(name) or any(char in name for char in "\r\n[]#*`<>"):
            raise ValueError(f"{path}: {label} must map to a plain, single-line name")
        aliases[label] = name
    return aliases
