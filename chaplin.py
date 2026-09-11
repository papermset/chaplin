import asyncio
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from ollama import AsyncClient
from pynput import keyboard

from benchmark.recorder import SessionRecorder
from benchmark.schema import load_manifest, validate_label
from decoding.reranker import NBestReranker
from decoding.session import InferenceSession


class Chaplin:
    def __init__(self, options=None, vsr_model=None):
        options = options or {}
        self.vsr_model = vsr_model
        self.recording = False
        self.res_factor = 3
        self.fps = 16
        self.frame_interval = 1 / self.fps
        self.frame_compression = 25
        self.nbest = options.get("nbest", 10)
        if type(self.nbest) is not int or self.nbest < 1:
            raise ValueError("nbest must be a positive integer")
        benchmark = options.get("benchmark", {})
        self.benchmark_enabled = benchmark.get("enabled", False)
        self.capture_logits = self.benchmark_enabled and options.get("capture_logits", False)
        self.type_output = not self.benchmark_enabled or benchmark.get("type_output", False)
        self.label_index = 0
        self.labels = None
        self.default_label = None
        self.recorder = None
        if self.benchmark_enabled:
            if benchmark.get("manifest"):
                self.labels = load_manifest(benchmark["manifest"])
            else:
                self.default_label = validate_label(
                    {
                        "speaker_id": benchmark.get("speaker_id", "unknown"),
                        "mode": benchmark.get("mode", "silent"),
                        "ground_truth": benchmark.get("ground_truth"),
                    }
                )
            self.recorder = SessionRecorder(
                benchmark.get("session_dir", "data/sessions/default"), options
            )
        rerank_options = options.get("reranker", {})
        self.ollama_client = AsyncClient()
        self.session = InferenceSession(
            NBestReranker(self.ollama_client, **rerank_options),
            options.get("context"),
            options.get("context_file"),
            options.get("audit_dir", "data/decoding"),
        )
        self.kbd_controller = keyboard.Controller()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.loop = asyncio.new_event_loop()
        self.async_thread = ThreadPoolExecutor(max_workers=1)
        self.async_thread.submit(self._run_event_loop)
        self.next_sequence_to_type = 0
        self.current_sequence = 0
        self.async_futures = []
        asyncio.run_coroutine_threadsafe(self._create_async_lock(), self.loop).result()
        # Option/Alt remains the only recording trigger.
        self.hotkey = keyboard.GlobalHotKeys({"<alt>": self.toggle_recording})
        self.hotkey.start()

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _create_async_lock(self):
        self.typing_condition = asyncio.Condition()

    def toggle_recording(self):
        self.recording = not self.recording

    def next_label(self):
        if self.labels is None:
            return self.default_label
        return self.labels[self.label_index] if self.label_index < len(self.labels) else None

    def show_prompt(self):
        if not self.benchmark_enabled:
            return
        label = self.next_label()
        if label is None:
            print("Benchmark manifest complete. Press q to exit.")
        else:
            print(
                f"Next sample: {label['speaker_id']} / {label['mode']} / {label['ground_truth']!r}"
            )

    async def correct_output_async(self, result, sequence_num, latency, submitted_at, sample=None):
        # Serialize reranks as well as typing, so context contains the preceding final output.
        async with self.typing_condition:
            await self.typing_condition.wait_for(lambda: self.next_sequence_to_type == sequence_num)
            try:
                text = await self.session.finish(result, latency, submitted_at, sample)
                if text and self.type_output:
                    self.kbd_controller.type(text + " ")
                return text
            except Exception as exc:
                if sample:
                    sample.update(status="output_failed", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                # A failed LLM, disk write or keyboard action must not block later samples.
                self.next_sequence_to_type += 1
                self.typing_condition.notify_all()

    def perform_inference(self, video_path, sample=None, submitted_at=None):
        result, latency, submitted_at = self.session.decode(
            self.vsr_model, video_path, sample, self.nbest, self.capture_logits, submitted_at
        )
        print(f"\nRAW OUTPUT: {result.best_text} ({len(result.hypotheses)} candidates)\n")
        sequence_num = self.current_sequence
        self.current_sequence += 1
        future = asyncio.run_coroutine_threadsafe(
            self.correct_output_async(result, sequence_num, latency, submitted_at, sample),
            self.loop,
        )
        self.async_futures.append(future)
        return {"output": result.best_text, "video_path": video_path}

    def start_webcam(self):
        cap, out, sample, output_path = None, None, None, None
        frame_count = 0
        started_at = None
        futures = []

        def finish_capture(aborted=False):
            nonlocal out, sample, output_path, frame_count
            if out is None:
                return
            out.release()
            out = None
            duration = time.perf_counter() - started_at
            if sample:
                sample.update(
                    camera={
                        **sample.metadata["camera"],
                        "frame_count": frame_count,
                        "capture_duration_s": duration,
                        "observed_fps": frame_count / duration if duration else 0,
                    },
                    status="aborted" if aborted else "captured",
                )
            if not aborted and frame_count >= self.fps * 2:
                futures.append(
                    self.executor.submit(
                        self.perform_inference, output_path, sample, time.perf_counter()
                    )
                )
            elif sample:
                sample.update(status="aborted" if aborted else "too_short")
            else:
                Path(output_path).unlink(missing_ok=True)
            sample, output_path, frame_count = None, None, 0
            self.show_prompt()

        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Cannot open camera 0")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640 // self.res_factor)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480 // self.res_factor)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            camera = {
                "device_index": 0,
                "fps": self.fps,
                "reported_fps": cap.get(cv2.CAP_PROP_FPS),
                "width": frame_width,
                "height": frame_height,
                "codec": "mp4v",
                "color": "grayscale",
                "jpeg_quality": self.frame_compression,
                "model_fps": getattr(self.vsr_model, "model_v_fps", None),
            }
            self.show_prompt()
            last_frame_time = 0
            while True:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if not self.recording and out is not None:
                    finish_capture()
                current_time = time.perf_counter()
                if current_time - last_frame_time >= self.frame_interval:
                    last_frame_time = current_time
                    ret, frame = cap.read()
                    if not ret:
                        raise RuntimeError("Camera frame read failed")
                    ok, buffer = cv2.imencode(
                        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.frame_compression]
                    )
                    if not ok:
                        raise RuntimeError("Frame compression failed")
                    compressed_frame = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
                    if self.recording:
                        if out is None:
                            if self.benchmark_enabled:
                                label = self.next_label()
                                if label is None:
                                    self.recording = False
                                    continue
                                sample = self.recorder.begin(label, camera)
                                output_path = sample.video_path
                                self.label_index += 1
                            else:
                                fd, output_path = tempfile.mkstemp(prefix="chaplin_", suffix=".mp4")
                                os.close(fd)
                            out = cv2.VideoWriter(
                                output_path,
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                self.fps,
                                (frame_width, frame_height),
                                False,
                            )
                            started_at = time.perf_counter()
                            if not out.isOpened():
                                raise RuntimeError("Cannot open video writer")
                        out.write(compressed_frame)
                        frame_count += 1
                        cv2.circle(compressed_frame, (frame_width - 20, 20), 10, (0, 0, 0), -1)
                    cv2.imshow("Chaplin", cv2.flip(compressed_frame, 1))
                # Inspect completed work without deleting files still in use by inference.
                for future in list(futures):
                    if future.done():
                        try:
                            future.result()
                        except Exception as exc:
                            print(f"Inference failed: {exc}")
                        futures.remove(future)
                for future in list(self.async_futures):
                    if future.done():
                        try:
                            future.result()
                        except Exception as exc:
                            print(f"Output failed: {exc}")
                        self.async_futures.remove(future)
        finally:
            self.hotkey.stop()
            try:
                finish_capture(aborted=True)
            finally:
                if cap is not None:
                    cap.release()
                cv2.destroyAllWindows()
                # Drain inference first, then pending LLM/audit/typing, then stop its loop.
                self.executor.shutdown(wait=True)
                for future in futures + self.async_futures:
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"Pending output failed: {exc}")
                self.loop.call_soon_threadsafe(self.loop.stop)
                self.async_thread.shutdown(wait=True)
                self.loop.close()
