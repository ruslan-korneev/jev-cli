"""The TypeSafe boundary: the official async SDK behind the runner's `Evaluator` protocol.

Retries live in the SDK (`RetryPolicy`: 408/429/5xx and connection errors, exponential backoff,
`retry-after`/`retry-after-ms` honoured; https://docs.typesafe.ai/sdk/python/api/retries.md).
Nothing outside this module imports `typesafe_sdk`.
"""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path

import httpx2
from pydantic import JsonValue
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    RetryPolicy,
    Score,
    ScoreAnswer,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAuthenticationError,
    TypeSafeError,
    TypeSafePermissionDeniedError,
)

from jev_kit.answers import Answer, noul_confidence
from jev_kit.errors import AuthError, UpstreamError
from jev_kit.runner import Evaluation, Evaluator
from jev_kit.spec import ChoiceQuestion, NoulQuestion, Question

__all__ = ["KEY_ENV", "TypeSafeEvaluator", "open_evaluator", "resolve_api_key"]

KEY_ENV = "TYPESAFE_API_KEY"
KEY_FILE = Path("~/.config/typesafe/api_key")

# Rate limits move ("adjusting dynamically", https://docs.typesafe.ai/models.md), so a 429 storm
# is expected under load: 6 retries with backoff up to 30 s, inside a 3-minute budget per item,
# ride out a minute-long squeeze without hanging a batch forever.
RETRY = RetryPolicy(max_retries=6, backoff_initial=1.0, backoff_max=30.0, timeout=180.0)
# Per HTTP attempt. The SDK default is 10 s; a 32k-token state with many questions needs more.
REQUEST_TIMEOUT_S = 60.0
_BODY_EXCERPT = 300


def resolve_api_key(env: Mapping[str, str], key_file: Path = KEY_FILE) -> str:
    key = env.get(KEY_ENV, "").strip()
    if key:
        return key
    path = key_file.expanduser()
    try:
        key = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        key = ""
    if not key:
        raise AuthError(f"no API key: set {KEY_ENV} or put the key in {key_file}")
    return key


class TypeSafeEvaluator:
    def __init__(self, client: AsyncTypeSafeClient) -> None:
        self._client = client

    async def evaluate(
        self, *, model: str, state: JsonValue, questions: Mapping[str, Question]
    ) -> Evaluation:
        if state is None or isinstance(state, bool | int | float):
            raise UpstreamError(f"state must be text, an object or an array, got {state!r}")
        try:
            response = await self._client.system_one(
                state,
                {qid: _sdk_question(q) for qid, q in questions.items()},
                model=model,
            )
        except (TypeSafeAuthenticationError, TypeSafePermissionDeniedError) as exc:
            raise AuthError(f"TypeSafe rejected the API key ({exc.status})") from exc
        except TypeSafeAPIError as exc:
            raise UpstreamError(_describe(exc), status=exc.status) from exc
        except TypeSafeAPIConnectionError as exc:
            raise UpstreamError(f"connection failed: {type(exc).__name__}") from exc
        except TypeSafeError as exc:
            raise UpstreamError(f"request refused by the SDK: {exc}") from exc

        missing = sorted(set(questions) - set(response.answers))
        if missing:
            # A partial response is not cached as if complete; the item fails and reruns.
            raise UpstreamError(f"response has no answer for {', '.join(missing)}")
        return Evaluation(
            model=response.model,
            answers={qid: _answer(response.answers[qid]) for qid in questions},
            input_tokens=response.usage.input_tokens or 0,
            output_tokens=response.usage.output_tokens or 0,
        )


@asynccontextmanager
async def open_evaluator(
    api_key: str,
    *,
    retry: RetryPolicy = RETRY,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> AsyncIterator[Evaluator]:
    try:
        client = AsyncTypeSafeClient(
            api_key=api_key, retry=retry, timeout=REQUEST_TIMEOUT_S, transport=transport
        )
    except TypeSafeError as exc:
        # The SDK validates the key's shape up front and keeps its value out of the message.
        raise AuthError(f"the API key is malformed: {exc}") from exc
    async with client:
        yield TypeSafeEvaluator(client)


def _sdk_question(question: Question) -> Noul | Choice | Score:
    body = question.model_dump(mode="json", exclude_none=True)
    if isinstance(question, NoulQuestion):
        return Noul.model_validate(body)
    if isinstance(question, ChoiceQuestion):
        return Choice.model_validate(body)
    return Score.model_validate(body)


def _answer(answer: NoulAnswer | ChoiceAnswer | ScoreAnswer) -> Answer:
    if isinstance(answer, NoulAnswer):
        return Answer(type="noul", noul=answer.noul, confidence=noul_confidence(answer.noul))
    if isinstance(answer, ChoiceAnswer):
        return Answer(
            type="choice",
            choice=answer.choice,
            probabilities=dict(answer.probabilities),
            confidence=answer.confidence,
        )
    return Answer(
        type="score",
        score=answer.score,
        probabilities={str(level): p for level, p in sorted(answer.probabilities.items())},
        confidence=answer.confidence,
    )


def _describe(exc: TypeSafeAPIError) -> str:
    body = exc.body
    detail = body if isinstance(body, str) else repr(body)
    if len(detail) > _BODY_EXCERPT:
        detail = detail[:_BODY_EXCERPT] + "…"
    return f"HTTP {exc.status}: {detail}"
