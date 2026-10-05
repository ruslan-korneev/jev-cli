"""Compare a run's answers with hand labels and suggest a threshold. Pure: no I/O, no network.

Labels are JSONL rows with the item id and the expected value under the question's id or under
`expected`: `{"id": 7, "is_ad": true}` or `{"id": 7, "expected": "news"}`.

- noul: accuracy / precision / recall / F1 at thresholds 0.05..0.95; recommends the best F1.
- choice: accuracy, per-option accuracy, and accuracy when only answers at or above a confidence
  gate are kept; recommends the lowest gate that reaches the target accuracy.
- score: exact and ±1-level accuracy on the rounded score, mean absolute error, and the same
  confidence gates.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import JsonValue

from jev_kit.errors import InputError

__all__ = [
    "ChoiceReport",
    "Gate",
    "NoulReport",
    "Report",
    "ScoreReport",
    "ThresholdRow",
    "calibrate",
]

NOUL_THRESHOLDS = tuple(round(0.05 * step, 2) for step in range(1, 20))
CONFIDENCE_GATES = tuple(round(0.1 * step, 1) for step in range(10))
_TRUE = {"true", "yes", "1"}
_FALSE = {"false", "no", "0"}


@dataclass(frozen=True, slots=True)
class ThresholdRow:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class Gate:
    min_confidence: float
    kept: int
    coverage: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class NoulReport:
    question: str
    matched: int
    positives: int
    thresholds: tuple[ThresholdRow, ...]
    recommended: ThresholdRow


@dataclass(frozen=True, slots=True)
class ChoiceReport:
    question: str
    matched: int
    accuracy: float
    per_option: dict[str, tuple[int, int]]  # expected option → (correct, labelled)
    gates: tuple[Gate, ...]
    target: float
    recommended: Gate | None


@dataclass(frozen=True, slots=True)
class ScoreReport:
    question: str
    matched: int
    exact: float
    within_one: float
    mae: float
    gates: tuple[Gate, ...]
    target: float
    recommended: Gate | None


type Report = NoulReport | ChoiceReport | ScoreReport

type Row = Mapping[str, JsonValue]


def calibrate(
    rows: Sequence[Row], labels: Sequence[Row], question: str, *, target: float = 0.9
) -> tuple[Report, dict[str, int]]:
    """The report, and counts of what could not be compared (and why)."""
    kind = _question_kind(rows, question)
    expected = _expected_by_id(labels, question)
    pairs: list[tuple[Row, JsonValue]] = []
    skipped = {"no_label": 0, "errored": 0}
    for row in rows:
        key = _id_key(row.get("id"))
        if key not in expected:
            skipped["no_label"] += 1
        else:
            label = expected.pop(key)
            if row.get(f"{question}.{kind}") is None:
                skipped["errored"] += 1
            else:
                pairs.append((row, label))
    skipped["label_without_row"] = len(expected)
    if not pairs:
        raise InputError(f"no item has both an answer to {question!r} and a label")
    if kind == "noul":
        return _noul(question, pairs), skipped
    if kind == "choice":
        return _choice(question, pairs, target), skipped
    return _score(question, pairs, target), skipped


def _question_kind(rows: Sequence[Row], question: str) -> str:
    for kind in ("noul", "choice", "score"):
        if any(f"{question}.{kind}" in row for row in rows):
            return kind
    known = sorted({key.split(".")[0] for row in rows for key in row if "." in key})
    raise InputError(f"no answers to {question!r} in the run output; questions: {known}")


def _id_key(value: JsonValue) -> str:
    # Ids may be 7 in one file and "7" in the other; compare them as text.
    return str(value)


def _expected_by_id(labels: Sequence[Row], question: str) -> dict[str, JsonValue]:
    found: dict[str, JsonValue] = {}
    for number, label in enumerate(labels, start=1):
        value = label.get(question, label.get("expected"))
        if value is None:
            raise InputError(f"label row {number} has neither {question!r} nor 'expected'")
        found[_id_key(label.get("id"))] = value
    return found


def _as_bool(value: JsonValue) -> bool:
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise InputError(f"a noul label must be true/false/yes/no/1/0, got {value!r}")


def _noul(question: str, pairs: list[tuple[Row, JsonValue]]) -> NoulReport:
    scored = [(_float(row[f"{question}.noul"]), _as_bool(label)) for row, label in pairs]
    table = tuple(_threshold_row(t, scored) for t in NOUL_THRESHOLDS)
    best = max(table, key=lambda r: (r.f1, r.accuracy, -abs(r.threshold - 0.5)))
    return NoulReport(
        question=question,
        matched=len(scored),
        positives=sum(1 for _, truth in scored if truth),
        thresholds=table,
        recommended=best,
    )


def _threshold_row(threshold: float, scored: list[tuple[float, bool]]) -> ThresholdRow:
    tp = sum(1 for p, truth in scored if p >= threshold and truth)
    fp = sum(1 for p, truth in scored if p >= threshold and not truth)
    fn = sum(1 for p, truth in scored if p < threshold and truth)
    tn = len(scored) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ThresholdRow(
        threshold=threshold,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        accuracy=(tp + tn) / len(scored),
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _choice(question: str, pairs: list[tuple[Row, JsonValue]], target: float) -> ChoiceReport:
    judged = [
        (str(row[f"{question}.choice"]) == str(label), _float(row[f"{question}.confidence"]))
        for row, label in pairs
    ]
    per_option: dict[str, tuple[int, int]] = {}
    for (_row, label), (correct, _) in zip(pairs, judged, strict=True):
        hits, total = per_option.get(str(label), (0, 0))
        per_option[str(label)] = (hits + int(correct), total + 1)
    gates = _gates(judged)
    return ChoiceReport(
        question=question,
        matched=len(judged),
        accuracy=sum(correct for correct, _ in judged) / len(judged),
        per_option=dict(sorted(per_option.items())),
        gates=gates,
        target=target,
        recommended=_first_reaching(gates, target),
    )


def _score(question: str, pairs: list[tuple[Row, JsonValue]], target: float) -> ScoreReport:
    levels: list[tuple[float, int, float]] = []
    for row, label in pairs:
        try:
            expected = int(str(label))
        except ValueError as exc:
            raise InputError(f"a score label must be a level number, got {label!r}") from exc
        levels.append(
            (_float(row[f"{question}.score"]), expected, _float(row[f"{question}.confidence"]))
        )
    rounded = [(math.floor(score + 0.5), expected) for score, expected, _ in levels]
    judged = [
        (got == expected, conf) for (got, expected), (*_, conf) in zip(rounded, levels, strict=True)
    ]
    gates = _gates(judged)
    return ScoreReport(
        question=question,
        matched=len(levels),
        exact=sum(got == expected for got, expected in rounded) / len(rounded),
        within_one=sum(abs(got - expected) <= 1 for got, expected in rounded) / len(rounded),
        mae=sum(abs(score - expected) for score, expected, _ in levels) / len(levels),
        gates=gates,
        target=target,
        recommended=_first_reaching(gates, target),
    )


def _gates(judged: list[tuple[bool, float]]) -> tuple[Gate, ...]:
    gates = []
    for gate in CONFIDENCE_GATES:
        kept = [correct for correct, conf in judged if conf >= gate]
        gates.append(
            Gate(
                min_confidence=gate,
                kept=len(kept),
                coverage=len(kept) / len(judged),
                accuracy=sum(kept) / len(kept) if kept else 0.0,
            )
        )
    return tuple(gates)


def _first_reaching(gates: tuple[Gate, ...], target: float) -> Gate | None:
    return next((g for g in gates if g.kept and g.accuracy >= target), None)


def _float(value: JsonValue) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InputError(f"expected a number in the run output, got {value!r}")
    return float(value)
