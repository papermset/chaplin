"""Exercise downloaded models on existing videos; never opens a live camera."""

import argparse
import asyncio
import json
import resource
import shutil
import sys
import time
import urllib.request
from pathlib import Path

from benchmark.recorder import SessionRecorder, atomic_json


def ollama_state():
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=5) as response:
            return json.load(response)
    except OSError as exc:
        return {"error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--config", default="configs/LRS3_V_WER19.1.ini")
    parser.add_argument("--session-dir", default="data/sessions/local_validation")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--nbest", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    for video in args.videos:
        if not Path(video).is_file():
            parser.error(f"Video not found: {video}")

    import cv2
    import torch
    from ollama import AsyncClient
    from decoding.reranker import NBestReranker
    from decoding.session import InferenceSession
    from pipelines.pipeline import InferencePipeline

    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available")
    camera = cv2.VideoCapture(args.videos[0])
    input_fps = camera.get(cv2.CAP_PROP_FPS)
    camera.release()
    if input_fps <= 0:
        parser.error("Cannot read video frame rate")
    started = time.perf_counter()
    pipeline = InferencePipeline(
        args.config,
        detector="mediapipe",
        face_track=True,
        device=args.device,
        input_v_fps=input_fps,
    )
    load_s = time.perf_counter() - started
    print(f"Loaded pretrained VSR + language model on {args.device} in {load_s:.2f}s", flush=True)
    recorder = SessionRecorder(args.session_dir, vars(args))
    session = InferenceSession(
        NBestReranker(AsyncClient(), timeout_s=args.timeout, enabled=not args.no_llm)
    )
    reports = []

    async def run():
        for video in args.videos:
            cap = cv2.VideoCapture(video)
            fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            metadata = {
                "fps": fps,
                "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                "frame_count": frames,
                "source": "existing_video",
                "model_fps": pipeline.model_v_fps,
            }
            cap.release()
            if abs(fps - input_fps) > 0.01:
                raise ValueError("Use a separate run for videos with different frame rates")
            sample = recorder.begin(
                {"mode": "normal", "speaker_id": "public_demo", "ground_truth": None}, metadata
            )
            shutil.copy2(video, sample.video_path)
            sample.update(
                status="captured", validation_only=True, source_video=str(Path(video).resolve())
            )
            decoded, latency, submitted = session.decode(
                pipeline, sample.video_path, sample, nbest=args.nbest, capture_logits=True
            )
            if args.device == "mps":
                torch.mps.synchronize()
            await session.finish(decoded, latency, submitted, sample)
            row = {
                "sample_id": sample.metadata["sample_id"],
                "video": str(video),
                "video_duration_s": frames / fps,
                "raw_vsr": decoded.best_text,
                "llm_output": sample.metadata["llm_output"],
                "latency": sample.metadata["latency"],
                "returned_nbest": len(decoded.hypotheses),
                "logits_shape": list(decoded.logits.shape) if decoded.logits is not None else None,
                "rerank_status": sample.metadata["rerank"]["status"],
                "rerank_error": sample.metadata["rerank"]["error"],
            }
            reports.append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
            atomic_json(
                Path(args.session_dir) / "runtime.json",
                {
                    "device": args.device,
                    "torch_threads": torch.get_num_threads(),
                    "model_load_s": load_s,
                    "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if sys.platform == "darwin" else 1024),
                    "ollama": ollama_state(),
                    "samples": reports,
                    "note": "Existing public video, not a live camera/silent-mouthing accuracy test. RSS and Ollama allocated sizes have different accounting.",
                },
            )

    asyncio.run(run())
    if not args.no_llm and any(row["rerank_status"] != "reranked" for row in reports):
        raise SystemExit(
            "Validation saved, but at least one live LLM rerank failed; inspect runtime.json"
        )


if __name__ == "__main__":
    main()
