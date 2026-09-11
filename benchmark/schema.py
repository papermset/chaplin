import json
from pathlib import Path

MODES = ("normal", "low_voice", "silent")
SCHEMA_VERSION = 1


def validate_label(label):
    if label.get("mode") not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not isinstance(label.get("speaker_id"), str) or not label["speaker_id"].strip():
        raise ValueError("speaker_id must be a nonempty string")
    if label.get("ground_truth") is not None and not isinstance(label["ground_truth"], str):
        raise ValueError("ground_truth must be a string or null (unlabelled)")
    return {key: label.get(key) for key in ("speaker_id", "mode", "ground_truth")}


def load_manifest(path):
    labels = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                labels.append(validate_label(json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    if not labels:
        raise ValueError("Manifest must contain at least one sample")
    return labels
