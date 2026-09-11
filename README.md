# Chaplin

This repository preserves the upstream Chaplin history and adds P0 benchmark recording
and P1 genuine N-best reranking. It is an experimental snapshot: local tests found no
reranking accuracy gain so far, and some samples remained decoded without a completed
LLM result. These issues remain open. Option/Alt is the only recording trigger.
Local recordings, model weights, environments and machine logs are excluded from Git;
links to those artifacts in the historical runtime reports work only in the original workspace.

![Chaplin Thumbnail](./thumbnail.png)

A visual speech recognition (VSR) tool that reads your lips in real-time and types whatever you silently mouth. Runs fully locally.

Relies on a [model](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages?tab=readme-ov-file#autoavsr-models) trained on the [Lip Reading Sentences 3](https://mmai.io/datasets/lip_reading/) dataset as part of the [Auto-AVSR](https://github.com/mpc001/auto_avsr) project.

Watch a demo of Chaplin [here](https://youtu.be/qlHi0As2alQ).

## Setup

1. Clone the repository, and `cd` into it:
   ```sh
   git clone https://github.com/papermset/chaplin
   cd chaplin
   ```
2. Run the setup script...
   ```sh
   ./setup.sh
   ```
   ...which will automatically download the required model files from Hugging Face Hub and place them in the appropriate directories:
   ```
   chaplin/
   ├── benchmarks/
       ├── LRS3/
           ├── language_models/
               ├── lm_en_subword/
           ├── models/
               ├── LRS3_V_WER19.1/
   ├── ...
   ```
3. Install and run `ollama`, and pull the [`qwen3:4b`](https://ollama.com/library/qwen3:4b) model.
4. Install [`uv`](https://github.com/astral-sh/uv).

## Usage

1. Run the following command:
   ```sh
   uv run --with-requirements requirements.txt --python 3.12 main.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe
   ```
2. Once the camera feed is displayed, you can start "recording" by pressing the `option` key (Mac) or the `alt` key (Windows/Linux), and start mouthing words.
3. To stop recording, press the `option` key (Mac) or the `alt` key (Windows/Linux) again. The raw VSR output will get logged in your terminal, and the LLM-corrected version will be typed at your cursor.
4. To exit gracefully, focus on the window displaying the camera feed and press `q`.

## P0: benchmark recording and evaluation

This fork is based on upstream commit `7aee1f8fca776ce4f63690063310b53573b7d804`.
Option/Alt toggles recording exactly as before and is the **only recording trigger**.
The camera window still uses `q` to exit. There is no automatic recording implementation,
configuration, or extension point.

Use Python 3.12. The original setup above is still required for pretrained weights and
Ollama. To install this fork and its tests into an isolated environment:

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

Start a labelled benchmark with the included three-mode example:

```sh
.venv/bin/python main.py \
  config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe \
  benchmark.enabled=true benchmark.session_dir=data/sessions/test_01 \
  benchmark.manifest=configs/benchmark_manifest.example.jsonl \
  context_file=configs/context.example.json capture_logits=true
```

The terminal shows the next sentence, speaker and mode **before** recording. Press
Option/Alt to start and again to stop each sample. Every newly opened recording consumes
one manifest row, including short or aborted clips. Once the manifest is exhausted,
further key presses do not create samples; press `q` and start another session with a
new manifest to collect more. Restarting the program starts the manifest at its first
row; existing samples are never overwritten. Use a new session directory for each run
if you do not want repeated recordings pooled in the same evaluation.

Alternatively supply one label to use for every clip:

```sh
.venv/bin/python main.py \
  config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe \
  benchmark.enabled=true benchmark.session_dir=data/sessions/silent_01 \
  benchmark.speaker_id=speaker_01 benchmark.mode=silent \
  'benchmark.ground_truth=Open the code editor'
```

`mode` is one of `normal`, `low_voice`, `silent`; it is a collection label, not an
inferred speech activity signal. `ground_truth` is the reference transcript. It can
be `null` for later manual annotation in `sample.json`. An empty string is an explicit
empty reference, distinct from an unlabelled sample. Ground truth is never passed to
the decoder or the LLM. Benchmark collection does not type by default; set
`benchmark.type_output=true` to enable typing during collection.

Each sample is written under `data/sessions/<session>/<unique-id>/`:

- `video.mp4`: retained for every benchmark clip, even short, failed or aborted clips.
- `sample.json`: schema version, run/sample IDs, timestamp, speaker, mode, ground truth,
  camera settings and measured capture duration/frame count, processing status,
  raw VSR, final LLM-selected text, full N-best scores, context, choice and latency.
- `ctc_logits.npz`: optional raw CTC activations and the vocabulary in column order;
  enabled with `capture_logits=true` in benchmark mode. Shape is `[frames, vocabulary]`.

A `run-<id>.json` stores the resolved run configuration. Paths inside each sample are
relative to its directory, so a session can be moved. JSON is replaced atomically.
`status` distinguishes `recording`, `captured`, `decoded`, `complete`, `too_short`,
`aborted`, `vsr_failed`, `output_failed` and `empty_beam`. A crash can leave a recording
or decoded record for inspection; these are excluded from paired evaluation unless
both outputs and a label exist. The existing minimum inference length remains two
seconds (32 captured frames). Pressing `q` during recording retains the partial
benchmark clip as aborted and does not run inference on it.

Normal mode still deletes temporary videos after VSR, including on inference failure.
It writes P1 text/score/choice audit JSON to `data/decoding/` (`audit_dir` is configurable).
It never deletes files by a broad filename prefix. Shutdown waits for queued inference
and bounded LLM calls before stopping background resources.

Latency is measured with a monotonic clock: `vsr_ms` includes video loading, landmarks,
preprocessing, encoder, beam search and optional logits extraction; `llm_ms` includes
LLM request/response validation; `vsr_queue_ms` and `llm_queue_ms` expose waiting.
`total_ms` runs from submission of a closed clip through rerank completion, including
queues and intermediate writes, but excludes recording, final audit write and typing.

Evaluate saved outputs repeatedly without a camera, model, network, or third-party
Python packages:

```sh
python3 -m benchmark.evaluate data/sessions/test_01
# Optional destination:
python3 -m benchmark.evaluate data/sessions/test_01 --output-dir data/reports/test_01
```

Outputs are `metrics.json`, `metrics.csv`, and `samples.csv`. They contain raw VSR WER/CER,
LLM-corrected WER/CER, aggregate and per-mode metrics, reference-unit/error counts,
sample rows, excluded-sample reasons, and rerank status counts. Here “LLM-corrected”
means the selected N-best text, including an explicitly marked raw fallback; it is
not an unconstrained rewrite. Both raw and LLM metrics use the same eligible samples.
Unlabelled and incomplete outputs are excluded visibly; an empty beam produces empty
outputs and is evaluated as deletions when a nonempty reference exists.

Normalization: Unicode NFKC, case folding, removal of Unicode punctuation, collapsed
whitespace. WER splits on whitespace; CER additionally removes spaces. WER is therefore
suited to whitespace-delimited languages; use CER for unsegmented Chinese. Aggregation
is micro averaging: sum edit distances / sum reference units, **not** an average of
sample WERs. Insertions on empty references contribute to corpus numerators. A zero
denominator is JSON `null` / blank CSV, not zero accuracy or NaN. Rates may exceed 1.
Offline evaluation recomputes metrics from saved predictions; it does not rerun models.

## P1: genuine N-best and constrained context reranking

The integration is at `InferencePipeline.decode()` → `AVSR.decode()` → the vendored
ESPnet `BatchBeamSearch`. Upstream's one-element slice and its one-best-only JSON
helper are bypassed in the structured path. The beam algorithm and pretrained weights
are unchanged. Existing `pipeline(video)` and `AVSR.infer(data)` still return a string.

```python
result = pipeline.decode(video_path, nbest=10, capture_logits=True)
print(result.best_text)
for hypothesis in result.hypotheses:
    print(hypothesis.candidate_id, hypothesis.text,
          hypothesis.score, hypothesis.scores, hypothesis.token_ids)
# result.logits: numpy array of pre-softmax CTC activations, or None with a reason
```

Live defaults are `nbest=10` and upstream `beam_size=40`. Request between 1 and the
configured beam size. Only genuine ended hypotheses are returned: if search returns
fewer than ten, fewer are saved and reranked. No candidate padding or LLM candidate
generation occurs. `requested_nbest`, `available_nbest` (all ended hypotheses), and
`returned_nbest` make shortages explicit. Duplicate text from different token paths is
preserved, as are the original token sequences, including SOS/EOS.

`score` is ESPnet's weighted cumulative beam score, with unweighted scorer components
in `scores` and weights in `scorer_weights` (attention decoder, CTC, language model,
length bonus where active). These are not calibrated probabilities or per-token logits.
Beam order is preserved. The actual search's max/min length settings are forwarded from
the INI file; short EOS candidates and search pruning remain upstream behavior.

The LLM receives all returned candidates, their visual scores, and only these context
fields (inline Hydra configuration or the JSON file shown above):

- `previous_text`: seeded from configuration, then the preceding final selected text.
- `current_app`: an explicitly supplied app name; no OS application inspection is needed.
- `custom_vocabulary`: list of user-supplied terms.
- `command_history`: seeded from configuration, then the last 20 finalized outputs;
  this is text context and does not execute commands.

The context file is reread before each rerank so app/vocabulary changes take effect.
After the first nonempty output, the session owns previous text/history so rereading
its original seed does not reset them. Reranks run in recording order on the background
loop, allowing the next sample to use the preceding final output while the camera
remains responsive. Benchmark ground truth and collection prompts are not context.

Ollama structured output must provide exactly one finite score in `[0, 1]` for every
candidate ID. The application selects the highest scored existing candidate, breaks
ties by beam order, and obtains text from its own candidate list. LLM-generated text
cannot become output. All raw scores, LLM heuristic `rerank_scores`, selected ID,
final text, context, model name, and fallback/error status are persisted.
The LLM scores are contextual judgments, not token likelihoods or calibrated confidence.
`think=false` disables Qwen reasoning for this bounded scoring task. The structured
schema restricts ratings to 0, 0.1, …, 1; ties retain beam order. Input includes
candidate ID, text and the original VSR total score; full scorer components and token
IDs remain in the sample audit. Requests use `num_ctx=4096`, `num_predict=1024`,
`temperature=0` and `seed=0`. These settings improve repeatability but do not guarantee identical model
responses across Ollama/model versions. Use `reranker.enabled=false` for raw selection
or `reranker.timeout_s=30` to change the default 60-second request timeout. Malformed
JSON, missing/duplicate/unknown IDs, invalid scores, network errors and timeouts fall
back to the original first hypothesis; missing rerank scores remain null.

CTC logits are extracted from `model.ctc.ctc_lo(enc_feats)` before softmax, reusing the
encoder pass. They are frame-level CTC activations, **not attention/LM token logits**.
ESPnet's attention scorer returns log-softmax vectors for each changing prefix, while
beam search retains only accumulated selected-path scores. It does not return a stable
full lattice or every branch's logits. This fork does not instrument all search states,
and does not mislabel CTC activations as those unavailable tensors. A model without a
CTC projection returns `logits=None` with `model_has_no_ctc_projection`; disabled
capture reports `capture_disabled`. Normal live mode does not persist large logits.

The live recording rate remains 16 FPS. Main passes that saved rate to preprocessing,
which resamples to the model's configured 25 FPS. This repeats frames rather than
recovering information; measured camera timing is stored for later domain analysis.
OpenCV now supplies RGB frames for preprocessing/detection because recent torchvision
versions removed `io.read_video`. Sharing the existing OpenCV backend also avoids
loading a second FFmpeg/AVFoundation binary from PyAV on macOS. MediaPipe is restricted to the legacy `solutions`
API used by the existing detector. Python 3.12 requires setuptools for vendored
ESPnet's distutils imports. The documented run path uses `detector=mediapipe`; optional
RetinaFace dependencies and GPU configurations were not validated in this fork.

## Tests and remaining validation

```sh
PYNPUT_BACKEND=dummy .venv/bin/python -m pytest -q
.venv/bin/ruff check benchmark decoding tests chaplin.py main.py \
  pipelines/model.py pipelines/pipeline.py pipelines/data/video_io.py \
  pipelines/data/data_module.py pipelines/detectors/mediapipe/detector.py \
  pipelines/detectors/retinaface/detector.py
```

Tests cover corpus/empty-reference metrics and repeatable CSV/JSON export; authentic
beam/token/component-score preservation; logits equality; small checkpoint loading,
video preprocessing and encoder/decoder integration; valid/invalid/timeout LLM replies;
context order; normal video cleanup; benchmark retention; short/aborted recordings;
and output-failure recovery. Global keys, camera and typing are simulated for lifecycle
tests. A real MP4 writer/reader is exercised separately. Neural smoke tests use small
random weights, not the downloaded LRS3 model. They verify interfaces and execution,
not recognition quality. See `IMPLEMENTATION_REPORT.md` for this run's exact results.

Full LRS3 + LM checkpoints and real Qwen inference have now passed on an M4 Mac
with 16 GB RAM: see [local runtime report](LOCAL_RUNTIME_REPORT.md). Still required:
a live camera/Option test with macOS accessibility permissions and labelled speaker
recordings to measure whether reranking actually improves WER. No quality/latency improvement is
claimed before those measurements. Changing context or reranker options can alter
results; retain the run configuration and exact model versions for comparisons.

## Local Mac validation (2026-09-11)

This workspace has the four pretrained files, Python environment and Ollama Qwen3:4B
installed. `setup.sh` now uses macOS curl and resumes interrupted downloads.
The local Ollama service was started with `brew services run ollama` (no login
registration); after restarting the machine, run that command again if needed.

Run an existing video without opening the camera or typing:

```sh
.venv/bin/python -m benchmark.check_local path/to/video.mp4 \
  --session-dir data/sessions/local_check --device cpu
```

The script records real N-best, CTC logits, Qwen scores and runtime/memory information.
Use `--no-llm` to measure raw VSR only. Videos in one invocation must share an FPS.
The supplied `--device mps` is a diagnostic path; it currently fails in the upstream
CTC prefix scorer's CPU/CUDA-only device selection. Live Mac usage stays on CPU.

Measured on two 3 s / 6 s example clips: CPU VSR 2.65 s / 4.09 s, Qwen rerank
11.93 s / 12.35 s, total 14.63 s / 16.53 s. These are individual measurements, not
latency guarantees or accuracy evidence. Both selected raw top-1. The examples are
unlabelled public speech footage; evaluation correctly excludes them from WER/CER.
29 tests pass. Provenance and the MPS failure log are under `validation/`.
