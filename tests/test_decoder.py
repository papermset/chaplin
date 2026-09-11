from configparser import ConfigParser
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from decoding.hypotheses import from_beam
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus
from pipelines.model import AVSR
from pipelines.pipeline import InferencePipeline


def small_avsr():
    """Real vendored attention decoder + CTC prefix scorer + beam, random weights."""
    torch.manual_seed(8)
    tokens = ["<blank>"] + [f"▁word{i}" for i in range(13)] + ["<eos>"]
    decoder = Decoder(
        len(tokens),
        attention_dim=8,
        attention_heads=2,
        linear_units=16,
        num_blocks=1,
        dropout_rate=0,
        positional_dropout_rate=0,
        self_attention_dropout_rate=0,
        src_attention_dropout_rate=0,
    )
    ctc = CTC(len(tokens), 8, 0, ctc_type="builtin")
    beam = BatchBeamSearch(
        beam_size=10,
        vocab_size=len(tokens),
        weights={"decoder": 0.9, "ctc": 0.1, "length_bonus": 0},
        scorers={
            "decoder": decoder,
            "ctc": CTCPrefixScorer(ctc, len(tokens) - 1),
            "length_bonus": LengthBonus(len(tokens)),
        },
        sos=len(tokens) - 1,
        eos=len(tokens) - 1,
        token_list=tokens,
        pre_beam_score_key="decoder",
    )
    model = AVSR.__new__(AVSR)
    torch.nn.Module.__init__(model)
    model.device = "cpu"
    model.token_list = tokens
    model.model = SimpleNamespace(encode=lambda data: data, ctc=ctc)
    model.beam_search = beam.eval()
    return model


def test_real_espnet_beam_and_ctc_logits():
    model = small_avsr()
    features = torch.randn(6, 8)
    decoded = model.decode(features, nbest=10, capture_logits=True)
    assert len(decoded.hypotheses) == 10
    assert decoded.available_nbest >= 10
    assert decoded.logits_kind == "ctc_pre_softmax"
    np.testing.assert_allclose(decoded.logits, model.model.ctc.ctc_lo(features).detach().numpy())
    assert decoded.logits.shape == (6, 15)
    assert [h.score for h in decoded.hypotheses] == sorted(
        [h.score for h in decoded.hypotheses], reverse=True
    )
    with torch.no_grad():
        raw = model.beam_search(features)
    for actual, saved in zip(raw, decoded.hypotheses):
        assert saved.token_ids == actual.yseq.tolist()
        assert saved.score == float(actual.score)
        assert saved.scores == {k: float(v) for k, v in actual.scores.items()}
        assert saved.score == pytest.approx(
            sum(saved.scores[k] * model.beam_search.weights[k] for k in saved.scores), abs=1e-5
        )
    assert model.infer(features) == decoded.best_text
    assert model.decode(features).logits_unavailable_reason == "capture_disabled"
    model.model.ctc = None
    assert (
        model.decode(features, capture_logits=True).logits_unavailable_reason
        == "model_has_no_ctc_projection"
    )
    with pytest.raises(ValueError, match="beam_size"):
        model.decode(features, nbest=11)


def test_pipeline_forwards_decode_options_and_legacy_string(tmp_path):
    pipeline = InferencePipeline.__new__(InferencePipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.maxlenratio, pipeline.minlenratio = 0.5, 0.1
    pipeline.process_landmarks = lambda filename, landmarks: None
    pipeline.dataloader = SimpleNamespace(load_data=lambda *args: "features")
    seen = []

    def decode(data, **kwargs):
        seen.append((data, kwargs))
        return from_beam([], [], kwargs["nbest"])

    pipeline.model = SimpleNamespace(decode=decode)
    video = tmp_path / "video.mp4"
    video.touch()
    assert pipeline.decode(str(video), nbest=7, capture_logits=True).requested_nbest == 7
    assert seen[0][1] == {
        "nbest": 7,
        "capture_logits": True,
        "maxlenratio": 0.5,
        "minlenratio": 0.1,
    }
    assert pipeline(str(video)) == ""


def test_config_and_video_temporal_resampling():
    from pipelines.data.transforms import VideoTransform

    config = ConfigParser()
    config.read("configs/LRS3_V_WER19.1.ini")
    assert config.get("input", "modality") == "video"
    output = VideoTransform(16 / 25)(torch.zeros(32, 96, 96))
    assert output.shape == (1, 50, 88, 88)


def test_video_to_nbest_with_checkpoint_roundtrip(tmp_path):
    """Full video path, with a small random checkpoint and supplied face landmarks."""
    import argparse
    import json
    import pickle
    import cv2
    from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
    from pipelines.detectors.mediapipe.video_process import VideoProcess

    args = E2E.add_arguments(argparse.ArgumentParser()).parse_args([])
    args.transformer_input_layer = "conv3d"
    args.adim, args.aheads, args.eunits, args.dunits = 8, 2, 16, 16
    args.elayers, args.dlayers = 1, 1
    args.mtlalpha, args.lsm_weight = 0.1, 0.1
    args.ctc_type, args.report_cer, args.report_wer = "builtin", False, False
    args.char_list = ["<blank>"] + [f"▁word{i}" for i in range(13)] + ["<eos>"]
    args.labels_type = "char"
    conf = tmp_path / "model.json"
    conf.write_text(json.dumps(vars(args)))
    model_path = tmp_path / "model.pth"
    torch.save(E2E(len(args.char_list), args).state_dict(), model_path)
    ini = tmp_path / "test.ini"
    ini.write_text(f"""[input]
modality=video
v_fps=16
[model]
v_fps=25
model_path={model_path}
model_conf={conf}
rnnlm=
rnnlm_conf=
[decode]
beam_size=10
penalty=0
ctc_weight=0.1
lm_weight=0
""")
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 16, (256, 256), False)
    assert writer.isOpened()
    for index in range(4):
        writer.write(np.full((256, 256), 50 + index * 30, dtype=np.uint8))
    writer.release()
    cropper = VideoProcess()
    points = cropper.get_stable_reference(cropper.reference, (256, 256), (256, 256))
    landmarks = tmp_path / "landmarks.pkl"
    with landmarks.open("wb") as stream:
        pickle.dump([points.copy() for _ in range(4)], stream)
    pipeline = InferencePipeline(str(ini), detector="mediapipe", face_track=False, device="cpu")
    decoded = pipeline.decode(str(video), str(landmarks), capture_logits=True)
    assert len(decoded.hypotheses) == 10
    assert decoded.logits.shape == (6, 15)  # 4 source frames at 16 FPS resampled to 25 FPS
    assert pipeline(str(video), str(landmarks)) == decoded.best_text
