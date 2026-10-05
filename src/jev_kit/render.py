"""Everything the CLI prints: output rows, progress, the run summary, and the human tables."""

import json
import os
import sys
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from pydantic import JsonValue
from rich.console import Console
from rich.table import Table

from jev_kit.answers import Answer, flat_answers
from jev_kit.calibrate import ChoiceReport, Gate, NoulReport, Report, ScoreReport
from jev_kit.plan import cost_usd
from jev_kit.runner import ItemResult

__all__ = [
    "Progress",
    "Summary",
    "calibration_text",
    "sample_table",
    "summarize",
    "to_row",
    "write_rows",
]

_TEXT_WIDTH = 70


def to_row(result: ItemResult) -> dict[str, JsonValue]:
    return {
        "id": result.id,
        "model": result.model,
        **flat_answers(result.answers),
        "usage": {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens},
        "cached_answers": result.cached_answers,
        "error": result.error,
    }


def write_rows(rows: Iterable[dict[str, JsonValue]], path: Path | None) -> None:
    """JSONL to stdout, or to `path` atomically (temp file in the same directory, then rename)."""
    lines = [json.dumps(row, ensure_ascii=False) + "\n" for row in rows]
    if path is None:
        sys.stdout.writelines(lines)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.writelines(lines)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class Summary:
    items: int
    ok: int
    failed: int
    requests: int
    cached_answers: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    seconds: float

    def line(self) -> str:
        return (
            f"done: {self.items} items ({self.ok} ok, {self.failed} failed) · "
            f"{self.requests} requests · {self.cached_answers} answers from cache · "
            f"{self.input_tokens:,} input tokens · ${self.cost_usd:.6f} · {self.seconds:.1f}s"
        )


def summarize(results: list[ItemResult], seconds: float) -> Summary:
    input_tokens = sum(r.input_tokens for r in results)
    failed = sum(1 for r in results if r.error)
    return Summary(
        items=len(results),
        ok=len(results) - failed,
        failed=failed,
        requests=sum(1 for r in results if r.requested),
        cached_answers=sum(r.cached_answers for r in results),
        input_tokens=input_tokens,
        output_tokens=sum(r.output_tokens for r in results),
        cost_usd=cost_usd(input_tokens),
        seconds=seconds,
    )


class Progress:
    """Item counter on stderr: one rewritten line on a terminal, a line per ~10% otherwise."""

    def __init__(self, total: int, stream: TextIO = sys.stderr) -> None:
        self._total = total
        self._stream = stream
        self._tty = stream.isatty()
        self._done = self._requests = self._failed = 0
        self._next_report = 0
        self._started = time.monotonic()

    def __call__(self, result: ItemResult) -> None:
        self._done += 1
        self._requests += result.requested
        self._failed += result.error is not None
        line = (
            f"[{self._done}/{self._total}] requests {self._requests} · failed {self._failed} · "
            f"{time.monotonic() - self._started:.0f}s"
        )
        if self._tty:
            self._stream.write(f"\r{line}")
            if self._done == self._total:
                self._stream.write("\n")
        elif self._done >= self._next_report or self._done == self._total:
            self._stream.write(line + "\n")
            self._next_report = self._done + max(1, self._total // 10)
        self._stream.flush()


def sample_table(results: list[ItemResult], questions: list[str], console: Console) -> None:
    table = Table(show_lines=True)
    table.add_column("id", no_wrap=True)
    table.add_column("state", max_width=_TEXT_WIDTH)
    for qid in questions:
        table.add_column(qid, no_wrap=True)
    for result in results:
        cells = [_cell(result.answers.get(qid)) for qid in questions]
        if result.error:
            cells = [f"error: {result.error}"] + [""] * (len(questions) - 1)
        table.add_row(str(result.id), _excerpt(result.state), *cells)
    console.print(table)


def _excerpt(state: JsonValue) -> str:
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= _TEXT_WIDTH * 2 else text[: _TEXT_WIDTH * 2 - 1] + "…"


def _cell(answer: Answer | None) -> str:
    if answer is None:
        return "—"
    if answer.type == "noul":
        return f"{answer.noul:.2f}"
    if answer.type == "choice":
        return f"{answer.choice}\nconf {answer.confidence:.2f}"
    return f"{answer.score:.2f}\nconf {answer.confidence:.2f}"


def calibration_text(report: Report, skipped: dict[str, int]) -> str:
    lines = [f"question {report.question}: {report.matched} labelled answers"]
    if any(skipped.values()):
        lines.append(
            "not compared: "
            + ", ".join(f"{count} {reason.replace('_', ' ')}" for reason, count in skipped.items())
        )
    if isinstance(report, NoulReport):
        lines.append(f"positives: {report.positives}")
        lines.append("threshold  acc    prec   recall f1     tp  fp  fn  tn")
        lines.extend(
            f"{r.threshold:>9.2f}  {r.accuracy:.3f}  {r.precision:.3f}  {r.recall:.3f}  "
            f"{r.f1:.3f}  {r.tp:>3} {r.fp:>3} {r.fn:>3} {r.tn:>3}"
            for r in report.thresholds
        )
        best = report.recommended
        lines.append(
            f"recommended: {report.question}.noul >= {best.threshold:.2f} "
            f"(f1 {best.f1:.3f}, precision {best.precision:.3f}, recall {best.recall:.3f})"
        )
        return "\n".join(lines)
    if isinstance(report, ChoiceReport):
        lines.append(f"accuracy: {report.accuracy:.3f}")
        lines.extend(
            f"  {option}: {hits}/{total}" for option, (hits, total) in report.per_option.items()
        )
    elif isinstance(report, ScoreReport):
        lines.append(
            f"exact (rounded score): {report.exact:.3f} · within one level: "
            f"{report.within_one:.3f} · mean abs error: {report.mae:.3f}"
        )
    lines.append("min conf  kept  coverage  accuracy")
    lines.extend(_gate_line(g) for g in report.gates)
    if report.recommended is None:
        lines.append(f"no confidence gate reaches accuracy {report.target:.2f}")
    else:
        g = report.recommended
        lines.append(
            f"recommended: keep {report.question}.confidence >= {g.min_confidence:.1f} "
            f"(accuracy {g.accuracy:.3f} on {g.coverage:.0%} of items); review the rest"
        )
    return "\n".join(lines)


def _gate_line(gate: Gate) -> str:
    return (
        f"{gate.min_confidence:>8.1f}  {gate.kept:>4}  {gate.coverage:>8.0%}  {gate.accuracy:.3f}"
    )
