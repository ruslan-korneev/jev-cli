"""The TypeSafe adapter over the real SDK, with HTTP faked by a mock transport.

Pins: parsing of a recorded real response, SDK retries on 429/529 honouring `retry-after-ms`,
and the translation of HTTP failures into jev-kit errors (auth → fatal, others → per item).
Responses in tests/fixtures/typesafe/ were recorded from api.typesafe.ai on 2026-10-03.
Backoff waits without a retry header are not exercised: they would sleep for real.
"""

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import httpx2
import pytest
from typesafe_sdk import RetryPolicy

from jev_kit.client import open_evaluator, resolve_api_key
from jev_kit.errors import AuthError, UpstreamError
from jev_kit.runner import Evaluation
from jev_kit.spec import Spec

FIXTURES = Path(__file__).parent.parent / "fixtures" / "typesafe"
SPEC = Spec.model_validate(
    {
        "state": "{text}",
        "questions": {
            "is_ad": {"type": "noul", "instructions": "Is this post an advertisement?"},
            "topic": {
                "type": "choice",
                "instructions": "Main topic?",
                "criteria": {"economy": None, "politics": None, "other": None},
            },
            "useful": {
                "type": "score",
                "instructions": "How actionable is it?",
                "criteria": ["Nothing to act on", "A fact without details", "Specific details"],
            },
        },
    }
)
FAKE_KEY = "ts-test-key"


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _evaluate(handler: Callable[[httpx2.Request], httpx2.Response]) -> Evaluation:
    async def go() -> Evaluation:
        async with open_evaluator(FAKE_KEY, transport=httpx2.MockTransport(handler)) as ev:
            return await ev.evaluate(model="jev-latest", state="text", questions=SPEC.questions)

    return asyncio.run(go())


def _ok(_: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, content=_fixture("system_one_ok.json"))


def test_recorded_response_becomes_typed_answers() -> None:
    evaluation = _evaluate(_ok)
    assert evaluation.model == "jev-1.13.0"
    assert (evaluation.input_tokens, evaluation.output_tokens) == (363, 76)
    ad, topic, useful = (evaluation.answers[q] for q in ("is_ad", "topic", "useful"))
    assert (ad.type, ad.noul) == ("noul", 0.03)
    assert ad.confidence == pytest.approx(0.94)
    assert (topic.choice, topic.confidence) == ("economy", 1.0)
    assert topic.probabilities == {"other": 0.0, "economy": 1.0, "politics": 0.0}
    assert (useful.score, useful.confidence) == (1.25, 0.6)
    assert useful.probabilities == {"0": 0.01, "1": 0.73, "2": 0.26}


def test_request_carries_bearer_key_model_and_questions() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return _ok(request)

    _evaluate(handler)
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/systemone"
    assert seen[0].headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert body["model"] == "jev-latest"
    assert body["state"] == "text"
    assert body["questions"]["topic"]["criteria"] == {
        "economy": None,
        "politics": None,
        "other": None,
    }


@pytest.mark.parametrize("status", [429, 529])
def test_rate_limit_and_overload_are_retried_after_the_server_hint(status: int) -> None:
    attempts: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        attempts.append(1)
        if len(attempts) < 3:
            return httpx2.Response(
                status, headers={"retry-after-ms": "0"}, json={"detail": "slow down"}
            )
        return _ok(request)

    assert _evaluate(handler).model == "jev-1.13.0"
    assert len(attempts) == 3


def test_bad_request_fails_the_item_with_status_and_detail() -> None:
    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(400, content=_fixture("system_one_400.json"))

    with pytest.raises(UpstreamError, match=r"HTTP 400: .*at most 10 levels") as caught:
        _evaluate(handler)
    assert caught.value.status == 400


def test_rejected_key_is_an_auth_error_without_the_key() -> None:
    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, content=_fixture("system_one_401.json"))

    with pytest.raises(AuthError, match=r"rejected the API key \(401\)") as caught:
        _evaluate(handler)
    assert FAKE_KEY not in str(caught.value)


def test_non_json_success_body_fails_the_item() -> None:
    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=b"<html>gateway</html>")

    with pytest.raises(UpstreamError):
        _evaluate(handler)


def test_response_missing_an_answer_fails_the_item() -> None:
    body = json.loads(_fixture("system_one_ok.json"))
    del body["answers"]["topic"]

    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=body)

    with pytest.raises(UpstreamError, match="no answer for topic"):
        _evaluate(handler)


def test_connection_failure_fails_the_item() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    attempts: list[int] = []

    def counting(request: httpx2.Request) -> httpx2.Response:
        attempts.append(1)
        return handler(request)

    async def go() -> None:
        # Retried like any connection error, minus the real backoff sleep.
        no_wait = RetryPolicy(max_retries=2, backoff_initial=0)
        transport = httpx2.MockTransport(counting)
        async with open_evaluator(FAKE_KEY, retry=no_wait, transport=transport) as ev:
            await ev.evaluate(model="jev-latest", state="t", questions=SPEC.questions)

    with pytest.raises(UpstreamError, match="connection failed: TypeSafeAPIConnectionError"):
        asyncio.run(go())
    assert len(attempts) == 3


@pytest.mark.parametrize(
    ("env", "file_text", "expected"),
    [
        ({"TYPESAFE_API_KEY": " env-key\n"}, "file-key", "env-key"),
        ({}, "file-key\n", "file-key"),
        ({"TYPESAFE_API_KEY": "  "}, "file-key", "file-key"),
    ],
)
def test_api_key_comes_from_env_then_key_file(
    tmp_path: Path, env: dict[str, str], file_text: str, expected: str
) -> None:
    key_file = tmp_path / "api_key"
    key_file.write_text(file_text, encoding="utf-8")
    assert resolve_api_key(env, key_file) == expected


def test_missing_api_key_names_both_places(tmp_path: Path) -> None:
    with pytest.raises(AuthError, match=r"set TYPESAFE_API_KEY or put the key in .*api_key"):
        resolve_api_key({}, tmp_path / "api_key")
