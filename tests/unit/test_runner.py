"""Plan + execute against a fake upstream: cache, resume, per-item failure, limits.

Pins: one request per item carrying only uncached questions; a rerun costs nothing; editing one
question re-asks only it; a failed item is recorded, not cached, and does not stop the run; a
rejected key stops the run. The real SDK and its retries are covered in test_client.py.
"""

import asyncio
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import JsonValue

from jev_kit.cache import AnswerCache
from jev_kit.errors import AuthError, InputError, UpstreamError
from jev_kit.items import Item
from jev_kit.plan import build_plan
from jev_kit.runner import Evaluation, ItemResult, Limits, execute
from jev_kit.spec import Question, Spec
from tests.unit.conftest import FakeEvaluator

LIMITS = Limits(concurrency=4, rps=1000)


def _spec(**overrides: dict[str, JsonValue]) -> Spec:
    questions: dict[str, JsonValue] = {
        "ad": {"type": "noul", "instructions": "Is it an ad?"},
        "topic": {
            "type": "choice",
            "instructions": "Topic?",
            "criteria": {"tech": None, "other": None},
        },
        "useful": {"type": "score", "instructions": "Useful?", "criteria": ["no", "yes"]},
    }
    questions.update(overrides)
    return Spec.model_validate({"state": "{text}", "questions": questions})


ITEMS = [Item(id=i, fields={"id": i, "text": f"post {i}"}) for i in range(5)]


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[AnswerCache]:
    store = AnswerCache(tmp_path / "answers.sqlite")
    yield store
    store.close()


def _run(
    spec: Spec, items: list[Item], evaluator: FakeEvaluator, cache: AnswerCache | None
) -> list[ItemResult]:
    return asyncio.run(execute(build_plan(spec, items, cache), evaluator, cache, LIMITS))


def test_each_item_is_one_request_with_every_question(cache: AnswerCache) -> None:
    fake = FakeEvaluator()
    results = _run(_spec(), ITEMS, fake, cache)
    assert sorted(str(state) for state, _ in fake.calls) == [f"post {i}" for i in range(5)]
    assert {questions for _, questions in fake.calls} == {("ad", "topic", "useful")}
    assert [r.id for r in results] == [0, 1, 2, 3, 4]
    assert all(r.requested and r.error is None and r.model == "jev-latest-1.0" for r in results)


def test_rerun_over_the_same_items_makes_no_request(cache: AnswerCache) -> None:
    _run(_spec(), ITEMS, FakeEvaluator(), cache)
    again = FakeEvaluator()
    results = _run(_spec(), ITEMS, again, cache)
    assert again.calls == []
    assert all(r.cached_answers == 3 and not r.requested for r in results)
    assert results[0].model == "jev-latest-1.0"
    assert list(results[0].answers) == ["ad", "topic", "useful"]


def test_editing_one_question_reasks_only_that_question(cache: AnswerCache) -> None:
    _run(_spec(), ITEMS, FakeEvaluator(), cache)
    edited = _spec(ad={"type": "noul", "instructions": "Is it a paid promotion?"})
    fake = FakeEvaluator()
    results = _run(edited, ITEMS, fake, cache)
    assert len(fake.calls) == 5
    assert {questions for _, questions in fake.calls} == {("ad",)}
    assert all(
        r.cached_answers == 2 and set(r.answers) == {"ad", "topic", "useful"} for r in results
    )
    # answers keep the questions.yaml order even when some came from the cache
    assert list(results[0].answers) == ["ad", "topic", "useful"]


def test_new_items_after_an_interrupted_run_are_the_only_requests(cache: AnswerCache) -> None:
    _run(_spec(), ITEMS[:3], FakeEvaluator(), cache)
    fake = FakeEvaluator()
    _run(_spec(), ITEMS, fake, cache)
    assert sorted(str(state) for state, _ in fake.calls) == ["post 3", "post 4"]


def test_a_failed_item_is_recorded_and_the_run_goes_on(cache: AnswerCache) -> None:
    fake = FakeEvaluator(failures={"post 2": UpstreamError("HTTP 400: too long", status=400)})
    results = _run(_spec(), ITEMS, fake, cache)
    failed = [r for r in results if r.error]
    assert [(r.id, r.error, r.answers) for r in failed] == [(2, "HTTP 400: too long", {})]
    assert sum(1 for r in results if r.error is None) == 4


def test_a_failed_item_is_not_cached_and_is_retried_on_rerun(cache: AnswerCache) -> None:
    _run(_spec(), ITEMS, FakeEvaluator(failures={"post 2": UpstreamError("boom")}), cache)
    fake = FakeEvaluator()
    results = _run(_spec(), ITEMS, fake, cache)
    assert [state for state, _ in fake.calls] == ["post 2"]
    assert all(r.error is None for r in results)


def test_rejected_key_stops_the_run(cache: AnswerCache) -> None:
    fake = FakeEvaluator(failures={f"post {i}": AuthError("rejected (401)") for i in range(5)})
    with pytest.raises(AuthError, match="rejected"):
        _run(_spec(), ITEMS, fake, cache)


def test_item_without_the_state_field_fails_without_a_request(cache: AnswerCache) -> None:
    items = [*ITEMS[:2], Item(id="media", fields={"id": "media", "caption": "photo"})]
    fake = FakeEvaluator()
    results = _run(_spec(), items, fake, cache)
    assert len(fake.calls) == 2
    assert results[2].error == "item has no text (fields: caption, id)"
    assert not results[2].requested


def test_template_typo_failing_every_item_is_refused_up_front(cache: AnswerCache) -> None:
    spec = Spec.model_validate(
        {"state": "{txt}", "questions": {"q": {"type": "noul", "instructions": "x"}}}
    )
    with pytest.raises(InputError, match="no item can build its state: item has no txt"):
        build_plan(spec, ITEMS, cache)


def test_without_a_cache_every_run_asks_again() -> None:
    _run(_spec(), ITEMS, FakeEvaluator(), None)
    fake = FakeEvaluator()
    _run(_spec(), ITEMS, fake, None)
    assert len(fake.calls) == 5


@dataclass
class _SlowEvaluator:
    in_flight: int = 0
    peak: int = 0
    inner: FakeEvaluator = field(default_factory=FakeEvaluator)

    async def evaluate(
        self, *, model: str, state: JsonValue, questions: Mapping[str, Question]
    ) -> Evaluation:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.in_flight -= 1
        return await self.inner.evaluate(model=model, state=state, questions=questions)


def test_requests_in_flight_never_exceed_the_concurrency_limit() -> None:
    slow = _SlowEvaluator()
    items = [Item(id=i, fields={"id": i, "text": f"p{i}"}) for i in range(20)]
    plan = build_plan(_spec(), items, None)
    asyncio.run(execute(plan, slow, None, Limits(concurrency=3, rps=1000)))
    assert slow.peak == 3


def test_corrupt_cache_row_is_a_miss_and_is_deleted(cache: AnswerCache) -> None:
    _run(_spec(), ITEMS[:1], FakeEvaluator(), cache)
    plan = build_plan(_spec(), ITEMS[:1], cache)
    key = plan.items[0].keys["ad"]
    cache._db.execute("UPDATE answers SET answer = '{broken' WHERE key = ?", (key,))  # noqa: SLF001
    cache._db.commit()  # noqa: SLF001
    assert key not in cache.get_many([key])
    fake = FakeEvaluator()
    _run(_spec(), ITEMS[:1], fake, cache)
    assert [questions for _, questions in fake.calls] == [("ad",)]
