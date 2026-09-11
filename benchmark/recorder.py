import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from benchmark.schema import SCHEMA_VERSION, validate_label


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Sample:
    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.metadata = metadata

    @property
    def video_path(self):
        return str(self.directory / "video.mp4")

    def update(self, **fields):
        self.metadata.update(fields)
        atomic_json(self.directory / "sample.json", self.metadata)

    def save_decode(self, result):
        decode = result.metadata()
        if result.logits is not None:
            import numpy as np

            path = self.directory / "ctc_logits.npz"
            np.savez_compressed(
                path, logits=result.logits, token_list=np.asarray(result.token_list)
            )
            decode["logits_path"] = path.name
            decode["logits_shape"] = list(result.logits.shape)
        self.update(raw_vsr=result.best_text, decode=decode, status="decoded")


class SessionRecorder:
    def __init__(self, directory, run_config=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex
        atomic_json(self.directory / f"run-{self.run_id}.json", run_config or {})

    def begin(self, label, camera):
        label = validate_label(label)
        sample_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
        )
        directory = self.directory / sample_id
        directory.mkdir()
        sample = Sample(
            directory,
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "run_id": self.run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                **label,
                "video_path": "video.mp4",
                "camera": dict(camera),
                "status": "recording",
                "raw_vsr": None,
                "llm_output": None,
                "latency": {},
            },
        )
        sample.update()
        return sample
