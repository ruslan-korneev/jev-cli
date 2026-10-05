"""Calibration metrics against hand labels.

Pins: the noul threshold table and its F1-based recommendation, choice/score accuracy and the
confidence gate, id matching across int/str, and the refusals. Does not judge whether F1 is the
right objective for a given task; the table is printed so a human can pick another row.
"""

import pytest
from pydantic import JsonValue

from jev_kit.calibrate import ChoiceReport, NoulReport, ScoreReport, calibrate
from jev_kit.errors import InputError


def _rows(
    question: str, kind: str, values: list[tuple[JsonValue, float]]
) -> list[dict[str, JsonValue]]:
    return [
        {"id": i, f"{question}.{kind}": value, f"{question}.confidence": conf}
        for i, (value, conf) in enumerate(values)
    ]


def test_noul_recommends_the_threshold_with_best_f1() -> None:
    probs = [0.95, 0.85, 0.60, 0.40, 0.30, 0.10]
    truth = [True, True, True, False, False, False]
    rows = _rows("ad", "noul", [(p, 0.0) for p in probs])
    labels: list[dict[str, JsonValue]] = [{"id": str(i), "ad": t} for i, t in enumerate(truth)]
    report, skipped = calibrate(rows, labels, "ad")
    assert isinstance(report, NoulReport)
    assert (report.matched, report.positives) == (6, 3)
    best = report.recommended
    # every threshold in (0.40, 0.60] separates perfectly; the tie goes to the one nearest 0.5
    assert (best.threshold, best.f1, best.accuracy) == (0.5, 1.0, 1.0)
    row_at_090 = next(r for r in report.thresholds if r.threshold == 0.9)
    assert (row_at_090.tp, row_at_090.fp, row_at_090.fn, row_at_090.tn) == (1, 0, 2, 3)
    assert skipped == {"no_label": 0, "errored": 0, "label_without_row": 0}


@pytest.mark.parametrize("label", [True, "yes", "Yes", 1, "1", "true"])
def test_noul_labels_accept_common_truthy_spellings(label: JsonValue) -> None:
    report, _ = calibrate(_rows("ad", "noul", [(0.9, 0.0)]), [{"id": 0, "expected": label}], "ad")
    assert isinstance(report, NoulReport)
    assert report.positives == 1


def test_choice_reports_accuracy_per_option_and_a_confidence_gate() -> None:
    answers: list[tuple[JsonValue, float]] = [
        ("news", 0.9),
        ("news", 0.8),
        ("ad", 0.95),
        ("news", 0.2),  # wrong, low confidence
    ]
    labels: list[dict[str, JsonValue]] = [
        {"id": 0, "expected": "news"},
        {"id": 1, "expected": "news"},
        {"id": 2, "expected": "ad"},
        {"id": 3, "expected": "ad"},
    ]
    report, _ = calibrate(_rows("topic", "choice", answers), labels, "topic", target=0.9)
    assert isinstance(report, ChoiceReport)
    assert report.accuracy == 0.75
    assert report.per_option == {"ad": (1, 2), "news": (2, 2)}
    assert report.recommended is not None
    assert (report.recommended.min_confidence, report.recommended.kept) == (0.3, 3)


def test_score_reports_exact_within_one_and_mae() -> None:
    answers: list[tuple[JsonValue, float]] = [(0.1, 0.9), (1.6, 0.5), (2.0, 1.0)]
    labels: list[dict[str, JsonValue]] = [
        {"id": 0, "expected": 0},
        {"id": 1, "expected": 1},
        {"id": 2, "expected": "0"},
    ]
    report, _ = calibrate(_rows("u", "score", answers), labels, "u")
    assert isinstance(report, ScoreReport)
    assert report.exact == pytest.approx(1 / 3)
    assert report.within_one == pytest.approx(2 / 3)
    assert report.mae == pytest.approx((0.1 + 0.6 + 2.0) / 3)


def test_errored_and_unlabelled_rows_are_counted_not_compared() -> None:
    rows: list[dict[str, JsonValue]] = [
        {"id": 0, "ad.noul": 0.9, "ad.confidence": 0.8},
        {"id": 1, "error": "HTTP 400"},
        {"id": 2, "ad.noul": 0.1, "ad.confidence": 0.8},
    ]
    labels: list[dict[str, JsonValue]] = [
        {"id": 0, "ad": True},
        {"id": 1, "ad": False},
        {"id": 9, "ad": True},
    ]
    report, skipped = calibrate(rows, labels, "ad")
    assert report.matched == 1
    assert skipped == {"no_label": 1, "errored": 1, "label_without_row": 1}


@pytest.mark.parametrize(
    ("rows", "labels", "message"),
    [
        ([{"id": 0, "x.noul": 0.5}], [{"id": 0, "ad": True}], r"no answers to 'ad'.*\['x'\]"),
        ([{"id": 0, "ad.noul": 0.5}], [{"id": 0}], "neither 'ad' nor 'expected'"),
        ([{"id": 0, "ad.noul": 0.5}], [{"id": 0, "ad": "maybe"}], "true/false/yes/no"),
        ([{"id": 0, "ad.noul": 0.5}], [{"id": 5, "ad": True}], "no item has both"),
    ],
)
def test_unusable_inputs_are_refused(
    rows: list[dict[str, JsonValue]], labels: list[dict[str, JsonValue]], message: str
) -> None:
    with pytest.raises(InputError, match=message):
        calibrate(rows, labels, "ad")
