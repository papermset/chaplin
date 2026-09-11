import json
import subprocess
import sys

import pytest

from benchmark.evaluate import evaluate
from benchmark.metrics import counts, distance, normalize, rates
from benchmark.recorder import SessionRecorder
from benchmark.schema import load_manifest


def test_edit_counts_and_normalization():
    assert normalize("  HELLO， world!  ") == "hello world"
    assert distance("kitten", "sitting") == 3
    assert counts("one two", "one three")["word_errors"] == 1
    assert counts("one two", "")["word_errors"] == 2
    assert counts("", "extra")["word_errors"] == 1
    assert rates(counts("", "extra"))["wer"] is None
    assert rates(counts("你好", "你号"))["cer"] == 0.5


def test_record_evaluate_repeatable(tmp_path):
    recorder = SessionRecorder(tmp_path, {"nbest": 10})
    for mode, truth, raw, llm in [
        ("normal", "one two three", "one two", "one two three"),
        ("silent", "four", "wrong", "four"),
        ("low_voice", "", "extra", ""),
    ]:
        sample = recorder.begin(
            {"mode": mode, "speaker_id": "s1", "ground_truth": truth}, {"fps": 16}
        )
        sample.update(raw_vsr=raw, llm_output=llm, status="complete")
    recorder.begin({"mode": "silent", "speaker_id": "s1", "ground_truth": None}, {})
    failed = recorder.begin({"mode": "silent", "speaker_id": "s1", "ground_truth": "hello"}, {})
    failed.update(status="vsr_failed")
    summary, rows = evaluate(tmp_path)
    assert len(rows) == 3
    assert len(summary["excluded_samples"]) == 2
    assert (
        summary["groups"]["all"]["raw"]["wer"] == 3 / 4
    )  # micro, includes insertion on empty reference
    assert summary["groups"]["all"]["llm"]["wer"] == 0
    assert summary["groups"]["low_voice"]["raw"]["wer"] is None
    assert summary["groups"]["silent"]["raw"]["wer"] == 1
    command = [sys.executable, "-m", "benchmark.evaluate", str(tmp_path)]
    subprocess.run(command, check=True, capture_output=True)
    first = {p.name: p.read_bytes() for p in (tmp_path / "evaluation").iterdir()}
    subprocess.run(command, check=True, capture_output=True)
    assert first == {p.name: p.read_bytes() for p in (tmp_path / "evaluation").iterdir()}
    assert set(first) == {"metrics.json", "metrics.csv", "samples.csv"}


def test_manifest_validation_and_unique_samples(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text('{"mode":"silent","speaker_id":"s","ground_truth":"hello"}\n')
    label = load_manifest(path)[0]
    recorder = SessionRecorder(tmp_path)
    assert recorder.begin(label, {}).directory != recorder.begin(label, {}).directory
    path.write_text('{"mode":"bad","speaker_id":"s"}\n')
    with pytest.raises(ValueError, match="mode"):
        load_manifest(path)
    path.write_text('{"mode":"silent","speaker_id":""}\n')
    with pytest.raises(ValueError, match="speaker_id"):
        load_manifest(path)
    assert json.loads((tmp_path / f"run-{recorder.run_id}.json").read_text()) == {}
