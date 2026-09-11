from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Hypothesis:
    candidate_id: int
    text: str
    score: float
    scores: dict[str, float]
    token_ids: list[int]


@dataclass
class DecodeResult:
    best_text: str
    hypotheses: list[Hypothesis]
    requested_nbest: int
    available_nbest: int
    scorer_weights: dict[str, float] = field(default_factory=dict)
    logits: Any = None
    logits_kind: str | None = None
    logits_unavailable_reason: str | None = None
    token_list: list[str] = field(default_factory=list)

    def metadata(self):
        # Do not deepcopy large tensors or decoder states into JSON.
        return {
            "best_text": self.best_text,
            "hypotheses": [asdict(h) for h in self.hypotheses],
            "requested_nbest": self.requested_nbest,
            "available_nbest": self.available_nbest,
            "returned_nbest": len(self.hypotheses),
            "scorer_weights": self.scorer_weights,
            "logits_kind": self.logits_kind,
            "logits_unavailable_reason": self.logits_unavailable_reason,
        }


def from_beam(hypotheses, token_list, nbest=10, scorer_weights=None):
    """Copy only actual ESPnet hypotheses, preserving beam order and raw scores."""
    if type(nbest) is not int or nbest < 1:
        raise ValueError("nbest must be a positive integer")
    results = []
    for index, hyp in enumerate(hypotheses[:nbest]):
        ids = [int(i) for i in hyp.yseq]
        # The first token is SOS; remove only known control symbols thereafter.
        tokens = [token_list[i] for i in ids[1:]]
        text = "".join(t for t in tokens if t not in {"<eos>", "<sos>", "<blank>"})
        text = text.replace("▁", " ").replace("<space>", " ").strip()
        results.append(
            Hypothesis(
                index, text, float(hyp.score), {k: float(v) for k, v in hyp.scores.items()}, ids
            )
        )
    return DecodeResult(
        results[0].text if results else "",
        results,
        nbest,
        len(hypotheses),
        dict(scorer_weights or {}),
        token_list=list(token_list),
    )
