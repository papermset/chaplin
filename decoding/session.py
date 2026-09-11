"""Inference bookkeeping shared by the live app and model-free smoke tests."""

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from benchmark.recorder import atomic_json
from decoding.reranker import Context


class InferenceSession:
    def __init__(self, reranker, context=None, context_file=None, audit_dir="data/decoding"):
        self.reranker = reranker
        self.context = Context.from_dict(context or {})
        self.context_file = context_file
        self.audit_dir = Path(audit_dir)
        self.has_output = False
        # Validate before opening a camera or starting background threads.
        self.snapshot_context()

    def snapshot_context(self):
        data = asdict(self.context)
        if self.context_file:
            supplied = json.loads(Path(self.context_file).read_text(encoding="utf-8"))
            Context.from_dict(supplied)
            if self.has_output:
                supplied.pop("previous_text", None)
                supplied.pop("command_history", None)
            data.update(supplied)
        return Context.from_dict(data)

    def decode(
        self, pipeline, video_path, sample=None, nbest=10, capture_logits=False, submitted_at=None
    ):
        submitted_at = submitted_at if submitted_at is not None else time.perf_counter()
        started = time.perf_counter()
        try:
            result = pipeline.decode(video_path, nbest=nbest, capture_logits=capture_logits)
            elapsed = (time.perf_counter() - started) * 1000
            if sample:
                sample.save_decode(result)
                sample.update(
                    latency={"vsr_ms": elapsed, "vsr_queue_ms": (started - submitted_at) * 1000}
                )
            return (
                result,
                {"vsr_ms": elapsed, "vsr_queue_ms": (started - submitted_at) * 1000},
                submitted_at,
            )
        except Exception as exc:
            if sample:
                sample.update(
                    status="vsr_failed",
                    error=f"{type(exc).__name__}: {exc}",
                    latency={
                        "vsr_ms": (time.perf_counter() - started) * 1000,
                        "total_ms": (time.perf_counter() - submitted_at) * 1000,
                    },
                )
            raise
        finally:
            if sample is None:
                Path(video_path).unlink(missing_ok=True)

    async def finish(self, result, latency, submitted_at, sample=None):
        context = self.snapshot_context()
        started = time.perf_counter()
        ranked = await self.reranker.rerank(result, context)
        latency = {
            **latency,
            "llm_ms": (time.perf_counter() - started) * 1000,
            "llm_queue_ms": (started - submitted_at) * 1000
            - latency["vsr_ms"]
            - latency["vsr_queue_ms"],
            "total_ms": (time.perf_counter() - submitted_at) * 1000,
        }
        fields = {
            "raw_vsr": result.best_text,
            "llm_output": ranked.final_text,
            "context": asdict(context),
            "rerank": ranked.metadata(),
            "latency": latency,
            "status": "complete" if result.hypotheses else "empty_beam",
            "llm_model": self.reranker.model,
        }
        if sample:
            sample.update(**fields)
        else:
            # Normal mode removes the temporary video but retains P1 score/choice audit.
            atomic_json(
                self.audit_dir / f"{uuid.uuid4().hex}.json",
                {"schema_version": 1, **fields, "decode": result.metadata()},
            )
        if ranked.final_text:
            self.context = Context(
                ranked.final_text,
                context.current_app,
                context.custom_vocabulary,
                (context.command_history + [ranked.final_text])[-20:],
            )
            self.has_output = True
        return ranked.final_text
