import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# No real global hooks or typing in tests.
os.environ["PYNPUT_BACKEND"] = "dummy"
import chaplin as app
from tests.test_reranker import result


@pytest.mark.parametrize(
    "benchmark,frames,stop,status",
    [
        (True, 36, True, "complete"),
        (True, 3, True, "too_short"),
        (True, 3, False, "aborted"),
        (False, 36, True, None),
        (False, 3, False, None),
    ],
)
def test_option_capture_lifecycle(tmp_path, monkeypatch, benchmark, frames, stop, status):
    class Hotkeys:
        def __init__(self, keys):
            assert list(keys) == ["<alt>"]
            self.keys = keys

        def start(self):
            pass

        def stop(self):
            pass

    typed = []
    monkeypatch.setattr(app.keyboard, "GlobalHotKeys", Hotkeys)
    monkeypatch.setattr(app.keyboard, "Controller", lambda: SimpleNamespace(type=typed.append))
    written = []

    class Writer:
        def __init__(self, path, *args):
            self.path = path
            written.append(path)
            Path(path).write_bytes(b"synthetic fixture")

        def isOpened(self):
            return True

        def write(self, frame):
            pass

        def release(self):
            pass

    class Camera:
        def isOpened(self):
            return True

        def set(self, *args):
            pass

        def get(self, prop):
            return 16 if prop == app.cv2.CAP_PROP_FPS else 96

        def read(self):
            return True, np.zeros((96, 96, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(app.cv2, "VideoCapture", lambda *args: Camera())
    monkeypatch.setattr(app.cv2, "VideoWriter", Writer)
    monkeypatch.setattr(app.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(app.cv2, "destroyAllWindows", lambda: None)
    options = {
        "benchmark": {
            "enabled": benchmark,
            "session_dir": str(tmp_path / "sessions"),
            "speaker_id": "test",
            "mode": "silent",
            "ground_truth": "OPEN THE CODE",
        },
        "reranker": {"enabled": False},
        "audit_dir": str(tmp_path / "audit"),
    }
    chaplin = app.Chaplin(options, SimpleNamespace(decode=lambda *args, **kwargs: result()))
    now = time.perf_counter()

    def tick():
        nonlocal now
        now += 0.07
        return now

    monkeypatch.setattr(app, "time", SimpleNamespace(perf_counter=tick))
    step = 0

    def key():
        nonlocal step
        step += 1
        if step == 1 or (stop and step == frames + 1):
            chaplin.hotkey.keys["<alt>"]()
        return ord("q") if step == frames + 2 else 0

    monkeypatch.setattr(app.cv2, "waitKey", lambda delay: key())
    chaplin.start_webcam()
    assert len(written) == 1
    assert Path(written[0]).exists() == benchmark
    if benchmark:
        data = json.loads(next((tmp_path / "sessions").rglob("sample.json")).read_text())
        assert data["status"] == status
        assert data["camera"]["frame_count"] >= frames
        assert not typed
    elif stop:
        assert typed == ["OPEN THE CODE "]
        assert len(list((tmp_path / "audit").glob("*.json"))) == 1


def test_failed_output_does_not_block_following_sequence():
    chaplin = app.Chaplin.__new__(app.Chaplin)
    chaplin.next_sequence_to_type = 0
    chaplin.type_output = False

    async def scenario():
        await chaplin._create_async_lock()
        calls = 0

        async def finish(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("disk error")
            return "next"

        chaplin.session = SimpleNamespace(finish=finish)
        first = chaplin.correct_output_async(None, 0, {}, 0)
        second = chaplin.correct_output_async(None, 1, {}, 0)
        results = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 1)
        assert isinstance(results[0], RuntimeError)
        assert results[1] == "next"
        assert chaplin.next_sequence_to_type == 2

    asyncio.run(scenario())
