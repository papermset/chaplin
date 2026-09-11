import asyncio
import json
from types import SimpleNamespace

import pytest

from decoding.hypotheses import from_beam
from decoding.reranker import Context, NBestReranker, parse_response


def result():
    return from_beam(
        [
            SimpleNamespace(yseq=[3, 1, 3], score=-3.12, scores={"decoder": -2.5}),
            SimpleNamespace(yseq=[3, 2, 3], score=-3.19, scores={"decoder": -2.6}),
        ],
        ["<blank>", "▁OPEN▁THE▁CODE", "▁OPEN▁THE▁CODEX", "<eos>"],
        10,
    )


def test_preserves_beam_never_fills_to_ten():
    decoded = result()
    assert decoded.requested_nbest == 10
    assert decoded.available_nbest == len(decoded.hypotheses) == 2
    assert decoded.best_text == "OPEN THE CODE"
    assert decoded.hypotheses[1].score == -3.19
    assert decoded.hypotheses[1].scores == {"decoder": -2.6}
    assert decoded.hypotheses[1].token_ids == [3, 2, 3]


def test_id_selection_cannot_inject_text_and_ties_keep_beam_order():
    content = json.dumps(
        {
            "scores": [{"candidate_id": 0, "score": 0.1}, {"candidate_id": 1, "score": 0.9}],
            "final_text": "invented by LLM",
        }
    )
    parsed = parse_response(content, result())
    assert parsed.selected_id == 1
    assert parsed.final_text == "OPEN THE CODEX"
    tie = '{"scores":[{"candidate_id":1,"score":0.5},{"candidate_id":0,"score":0.5}]}'
    assert parse_response(tie, result()).selected_id == 0


@pytest.mark.parametrize(
    "content",
    [
        '{"scores":[{"candidate_id":9,"score":0.5}]}',
        '{"scores":[{"candidate_id":0,"score":0.5}]}',
        '{"scores":[{"candidate_id":0,"score":0.5},{"candidate_id":0,"score":0.5}]}',
        '{"scores":[{"candidate_id":0,"score":NaN},{"candidate_id":1,"score":0.5}]}',
        '{"scores":[{"candidate_id":true,"score":0.5}]}',
        '{"scores":[{"candidate_id":0,"score":true}]}',
        '{"scores":[{"candidate_id":0,"score":2}]}',
        "not json",
    ],
)
def test_bad_responses_fall_back(content):
    class Client:
        async def chat(self, **kwargs):
            return {"message": {"content": content}}

    ranked = asyncio.run(NBestReranker(Client()).rerank(result(), Context()))
    assert ranked.status == "fallback"
    assert ranked.final_text == result().best_text
    assert all(s is None for s in ranked.rerank_scores.values())
    assert ranked.response == content


def test_live_request_contract_and_valid_selection():
    class Client:
        async def chat(self, **kwargs):
            payload = json.loads(kwargs["messages"][1]["content"])
            assert [h["vsr_score"] for h in payload["candidates"]] == [-3.12, -3.19]
            schema = kwargs["format"]["properties"]["scores"]["items"]["properties"]["score"]
            assert min(schema["enum"]) == 0 and max(schema["enum"]) == 1
            return {
                "message": {
                    "content": '{"scores":[{"candidate_id":0,"score":0.2},{"candidate_id":1,"score":0.9}]}'
                }
            }

    ranked = asyncio.run(NBestReranker(Client()).rerank(result(), Context()))
    assert ranked.status == "reranked"
    assert ranked.final_text == "OPEN THE CODEX"


def test_timeout_context_and_empty_beam():
    class Client:
        async def chat(self, **kwargs):
            self.kwargs = kwargs
            await asyncio.sleep(1)

    client = Client()
    context = Context("hello", "editor", ["Codex"], ["open"])
    ranked = asyncio.run(NBestReranker(client, timeout_s=0.001).rerank(result(), context))
    assert ranked.status == "fallback" and "TimeoutError" in ranked.error
    assert client.kwargs["think"] is False
    assert client.kwargs["options"]["num_ctx"] == 4096
    assert client.kwargs["options"]["num_predict"] == 1024
    payload = json.loads(client.kwargs["messages"][1]["content"])
    assert set(payload["context"]) == {
        "previous_text",
        "current_app",
        "custom_vocabulary",
        "command_history",
    }
    empty = from_beam([], [], 10)
    assert asyncio.run(NBestReranker(None).rerank(empty, Context())).status == "empty_beam"
