"""Shared fakes: an Evaluator that records every request and answers deterministically."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic import JsonValue

from jev_kit.answers import Answer
from jev_kit.errors import JevError
from jev_kit.runner import Evaluation
from jev_kit.spec import ChoiceQuestion, NoulQuestion, Question


@dataclass
class FakeEvaluator:
    """Answers every question; fails for states listed in `failures` with the given error."""

    failures: dict[str, JevError] = field(default_factory=dict)
    calls: list[tuple[JsonValue, tuple[str, ...]]] = field(default_factory=list)
    tokens_per_call: int = 100

    async def evaluate(
        self, *, model: str, state: JsonValue, questions: Mapping[str, Question]
    ) -> Evaluation:
        self.calls.append((state, tuple(questions)))
        if isinstance(state, str) and state in self.failures:
            raise self.failures[state]
        return Evaluation(
            model=f"{model}-1.0",
            answers={qid: _answer(q) for qid, q in questions.items()},
            input_tokens=self.tokens_per_call,
            output_tokens=10,
        )


def _answer(question: Question) -> Answer:
    if isinstance(question, NoulQuestion):
        return Answer(type="noul", noul=0.9, confidence=0.8)
    if isinstance(question, ChoiceQuestion):
        first = next(iter(question.criteria))
        return Answer(type="choice", choice=first, probabilities={first: 1.0}, confidence=1.0)
    return Answer(type="score", score=1.0, probabilities={"0": 0.0, "1": 1.0}, confidence=1.0)
