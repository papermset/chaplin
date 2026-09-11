import asyncio
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.recorder import SessionRecorder
from decoding.reranker import NBestReranker
from decoding.session import InferenceSession
from tests.test_reranker import result


def sample_in(path):
    return SessionRecorder(path).begin(
        {"mode": "silent", "speaker_id": "test", "ground_truth": "OPEN THE CODEX"}, {"fps": 16}
    )


def test_benchmark_retention_logits_and_context(tmp_path):
    sample = sample_in(tmp_path)
    from pathlib import Path

    Path(sample.video_path).write_bytes(b"test clip")
    decoded = result()
    decoded.logits = np.zeros((4, 4), dtype=np.float32)
    decoded.logits_kind = "ctc_pre_softmax"

    class Client:
        async def chat(self, **kwargs):
            return {
                "message": {
                    "content": '{"scores":[{"candidate_id":0,"score":0.1},{"candidate_id":1,"score":0.9}]}'
                }
            }

    session = InferenceSession(NBestReranker(Client()), context={"current_app": "editor"})
    pipeline = SimpleNamespace(decode=lambda *args, **kwargs: decoded)
    result_data, latency, submitted = session.decode(
        pipeline, sample.video_path, sample, capture_logits=True
    )
    assert Path(sample.video_path).exists()
    assert (sample.directory / "ctc_logits.npz").exists()
    output = asyncio.run(session.finish(result_data, latency, submitted, sample))
    data = json.loads((sample.directory / "sample.json").read_text())
    assert output == data["llm_output"] == "OPEN THE CODEX"
    assert data["rerank"]["selected_id"] == 1
    assert data["decode"]["hypotheses"][0]["score"] == -3.12
    assert all(data["latency"][key] >= 0 for key in ("vsr_ms", "llm_ms", "total_ms"))
    assert session.snapshot_context().previous_text == output
    assert session.snapshot_context().command_history == [output]
    assert "ground_truth" not in data["context"]


def test_normal_cleanup_and_failure_retention(tmp_path):
    session = InferenceSession(NBestReranker(None, enabled=False), audit_dir=tmp_path / "audit")
    video = tmp_path / "temporary.mp4"
    video.touch()
    pipeline = SimpleNamespace(decode=lambda *args, **kwargs: result())
    data, latency, submitted = session.decode(pipeline, str(video))
    assert not video.exists()
    asyncio.run(session.finish(data, latency, submitted))
    assert len(list((tmp_path / "audit").glob("*.json"))) == 1

    def fail(*args, **kwargs):
        raise RuntimeError("no face")

    pipeline.decode = fail
    video.touch()
    with pytest.raises(RuntimeError):
        session.decode(pipeline, str(video))
    assert not video.exists()
    sample = sample_in(tmp_path / "benchmark")
    from pathlib import Path

    Path(sample.video_path).touch()
    with pytest.raises(RuntimeError):
        session.decode(pipeline, sample.video_path, sample)
    assert Path(sample.video_path).exists()
    assert sample.metadata["status"] == "vsr_failed"


def test_context_file_updates_without_resetting_history(tmp_path):
    path = tmp_path / "context.json"
    path.write_text('{"previous_text":"seed","current_app":"editor","command_history":["before"]}')
    session = InferenceSession(
        NBestReranker(None, enabled=False), context_file=str(path), audit_dir=tmp_path / "audit"
    )
    assert session.snapshot_context().previous_text == "seed"
    asyncio.run(session.finish(result(), {"vsr_ms": 0, "vsr_queue_ms": 0}, time.perf_counter()))
    path.write_text(
        '{"previous_text":"reset","current_app":"terminal","custom_vocabulary":["new"]}'
    )
    context = session.snapshot_context()
    assert context.previous_text == "OPEN THE CODE"
    assert context.current_app == "terminal"
    assert context.command_history == ["before", "OPEN THE CODE"]
