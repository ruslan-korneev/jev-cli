"""Execute a plan: one request per item with every unanswered question, under two limits.

The concurrency semaphore caps requests in flight; the rate limiter caps requests started per
second. They are different limits: slow responses fill the first, fast ones hit the second.

An item's failure (after the SDK's retries) is recorded on its row and the run goes on. A
rejected API key stops the run, because every later request would fail the same way.
"""

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol

from aiolimiter import AsyncLimiter
from pydantic import JsonValue

from jev_kit.answers import Answer
from jev_kit.cache import AnswerCache, CachedAnswer
from jev_kit.errors import AuthError, JevError
from jev_kit.items import ItemId
from jev_kit.plan import ItemPlan, RunPlan
from jev_kit.spec import Question

__all__ = ["Evaluation", "Evaluator", "ItemResult", "Limits", "execute"]


@dataclass(frozen=True, slots=True)
class Evaluation:
    model: str
    answers: dict[str, Answer]
    input_tokens: int
    output_tokens: int


class Evaluator(Protocol):
    async def evaluate(
        self, *, model: str, state: JsonValue, questions: Mapping[str, Question]
    ) -> Evaluation: ...


@dataclass(frozen=True, slots=True)
class Limits:
    concurrency: int
    rps: float


@dataclass(frozen=True, slots=True)
class ItemResult:
    id: ItemId
    state: JsonValue
    answers: dict[str, Answer]
    model: str | None
    requested: bool
    cached_answers: int
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


async def execute(
    plan: RunPlan,
    evaluator: Evaluator,
    cache: AnswerCache | None,
    limits: Limits,
    on_done: Callable[[ItemResult], None] = lambda _: None,
) -> list[ItemResult]:
    """Results in plan order. Raises AuthError if the key is rejected."""
    semaphore = asyncio.Semaphore(limits.concurrency)
    limiter = AsyncLimiter(max_rate=limits.rps, time_period=1.0)
    order = list(plan.spec.questions)

    async def run_one(item: ItemPlan) -> ItemResult:
        result = await _run_item(item)
        on_done(result)
        return result

    async def _run_item(item: ItemPlan) -> ItemResult:
        cached = {qid: c.answer for qid, c in item.cached.items()}
        base = ItemResult(
            id=item.id,
            state=item.state,
            answers=cached,
            model=next((c.model for c in item.cached.values()), None),
            requested=False,
            cached_answers=len(cached),
            error=item.error,
        )
        if item.error is not None or not item.to_ask:
            return _ordered(base, order)
        try:
            async with semaphore, limiter:
                evaluation = await evaluator.evaluate(
                    model=plan.spec.model,
                    state=item.state,
                    questions={qid: plan.spec.questions[qid] for qid in item.to_ask},
                )
        except AuthError:
            raise
        except JevError as exc:
            return _ordered(replace(base, requested=True, error=str(exc)), order)
        if cache is not None:
            cache.put_many(
                (item.keys[qid], CachedAnswer(model=evaluation.model, answer=answer))
                for qid, answer in evaluation.answers.items()
            )
        return _ordered(
            replace(
                base,
                answers={**cached, **evaluation.answers},
                model=evaluation.model,
                requested=True,
                input_tokens=evaluation.input_tokens,
                output_tokens=evaluation.output_tokens,
            ),
            order,
        )

    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(run_one(item)) for item in plan.items]
    except* AuthError as group_error:
        raise group_error.exceptions[0] from None
    return [task.result() for task in tasks]


def _ordered(result: ItemResult, order: list[str]) -> ItemResult:
    answers = {qid: result.answers[qid] for qid in order if qid in result.answers}
    return replace(result, answers=answers)
