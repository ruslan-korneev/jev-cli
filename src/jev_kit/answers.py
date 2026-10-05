"""One answer, as cached and as written: the flat row format of `jev run`.

Row keys (the output contract, also in README "Output"): `id`, `model`, then per question
`<q>.noul` | `<q>.choice` | `<q>.score`, `<q>.confidence`, `<q>.probabilities` (choice and
score), then `usage`, `cached_answers` and `error`.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

__all__ = ["Answer", "flat_answers", "noul_confidence"]


class Answer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["noul", "choice", "score"]
    noul: float | None = None
    choice: str | None = None
    score: float | None = None
    # Choice: option → p. Score: level index as a string ("0", "1", …) → p, like the API.
    probabilities: dict[str, float] | None = None
    confidence: float


def noul_confidence(p: float) -> float:
    """|2p - 1|. The API returns no confidence for a noul; this is the formula the docs give
    (https://docs.typesafe.ai/confidence.md#noul) so every question has the same column."""
    return abs(2 * p - 1)


def flat_answers(answers: dict[str, Answer]) -> dict[str, JsonValue]:
    row: dict[str, JsonValue] = {}
    for qid, answer in answers.items():
        value: JsonValue = {"noul": answer.noul, "choice": answer.choice, "score": answer.score}[
            answer.type
        ]
        row[f"{qid}.{answer.type}"] = value
        row[f"{qid}.confidence"] = answer.confidence
        if answer.probabilities is not None:
            row[f"{qid}.probabilities"] = dict(answer.probabilities)
    return row
