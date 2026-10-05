"""The `jev` commands end to end, with the upstream replaced by the recording fake.

Pins: output rows and exit codes, a second run costing zero requests, --dry-run touching neither
the network nor the key, the sample table, and calibrate's text. The real HTTP path is in
test_client.py; the live check is in README "Verification".
"""

import json
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jev_kit.cli import build_app
from jev_kit.errors import UpstreamError
from jev_kit.runner import Evaluator
from tests.unit.conftest import FakeEvaluator

QUESTIONS = """\
state: "{text}"
questions:
  ad:
    type: noul
    instructions: Is it an ad?
  topic:
    type: choice
    instructions: Topic?
    criteria: {tech: null, other: null}
"""


@pytest.fixture
def files(tmp_path: Path) -> Path:
    (tmp_path / "q.yaml").write_text(QUESTIONS, encoding="utf-8")
    lines = [json.dumps({"id": i, "text": f"post {i}"}) for i in range(4)]
    (tmp_path / "items.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


class _Factory:
    def __init__(self, fake: FakeEvaluator) -> None:
        self.fake = fake
        self.opened = 0

    def __call__(self) -> AbstractAsyncContextManager[Evaluator]:
        @asynccontextmanager
        async def opened() -> AsyncIterator[Evaluator]:
            self.opened += 1
            yield self.fake

        return opened()


def _invoke(factory: _Factory, files: Path, *args: str) -> tuple[int, str, str]:
    runner = CliRunner()
    env = {"JEV_KIT_CACHE": str(files / "cache.sqlite")}
    result = runner.invoke(build_app(factory), list(args), env=env)
    return result.exit_code, result.stdout, result.stderr


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_run_writes_flat_rows_and_a_summary(files: Path) -> None:
    factory = _Factory(FakeEvaluator())
    out = files / "out.jsonl"
    code, _, err = _invoke(
        factory, files, "run", str(files / "q.yaml"), str(files / "items.jsonl"), "-o", str(out)
    )
    assert code == 0
    rows = _rows(out)
    assert [r["id"] for r in rows] == [0, 1, 2, 3]
    assert rows[0] == {
        "id": 0,
        "model": "jev-latest-1.0",
        "ad.noul": 0.9,
        "ad.confidence": 0.8,
        "topic.choice": "tech",
        "topic.confidence": 1.0,
        "topic.probabilities": {"tech": 1.0},
        "usage": {"input_tokens": 100, "output_tokens": 10},
        "cached_answers": 0,
        "error": None,
    }
    assert (
        "done: 4 items (4 ok, 0 failed) · 4 requests · 0 answers from cache · 400 input tokens"
        in err
    )


def test_second_run_makes_zero_requests_and_needs_no_client(files: Path) -> None:
    args = ("run", str(files / "q.yaml"), str(files / "items.jsonl"), "-o", str(files / "o.jsonl"))
    _invoke(_Factory(FakeEvaluator()), files, *args)
    again = _Factory(FakeEvaluator())
    code, _, err = _invoke(again, files, *args)
    assert code == 0
    assert again.fake.calls == []
    assert again.opened == 0  # no key needed when everything is cached
    assert "· 0 requests · 8 answers from cache · 0 input tokens" in err
    assert _rows(files / "o.jsonl")[0]["cached_answers"] == 2


def test_failed_item_exits_4_and_keeps_the_other_rows(files: Path) -> None:
    fake = FakeEvaluator(failures={"post 1": UpstreamError("HTTP 400: too long", status=400)})
    out = files / "out.jsonl"
    code, _, err = _invoke(
        _Factory(fake),
        files,
        "run",
        str(files / "q.yaml"),
        str(files / "items.jsonl"),
        "-o",
        str(out),
    )
    assert code == 4
    rows = _rows(out)
    assert rows[1]["error"] == "HTTP 400: too long"
    assert "ad.noul" not in rows[1]
    assert [r["error"] for r in rows if r["id"] != 1] == [None, None, None]
    assert "1 items failed" in err


def test_limit_takes_the_first_items(files: Path) -> None:
    fake = FakeEvaluator()
    code, out, _ = _invoke(
        _Factory(fake),
        files,
        "run",
        str(files / "q.yaml"),
        str(files / "items.jsonl"),
        "--limit",
        "2",
    )
    assert code == 0
    assert [json.loads(line)["id"] for line in out.splitlines()] == [0, 1]


def test_dry_run_prints_the_first_payload_without_opening_a_client(files: Path) -> None:
    factory = _Factory(FakeEvaluator())
    code, out, _ = _invoke(
        factory, files, "run", str(files / "q.yaml"), str(files / "items.jsonl"), "--dry-run"
    )
    assert code == 0
    plan = json.loads(out)
    assert factory.opened == 0
    assert (plan["items"], plan["requests"], plan["cached_answers"]) == (4, 4, 0)
    assert plan["first_request"] == {
        "state": "post 0",
        "model": "jev-latest",
        "questions": {
            "ad": {"type": "noul", "instructions": "Is it an ad?"},
            "topic": {
                "type": "choice",
                "instructions": "Topic?",
                "criteria": {"tech": None, "other": None},
            },
        },
    }
    assert plan["estimated_input_tokens"] > 0


def test_bad_questions_file_exits_2_with_the_reason(files: Path) -> None:
    (files / "bad.yaml").write_text("state: '{text}'\nquestions: {}\n", encoding="utf-8")
    code, _, err = _invoke(
        _Factory(FakeEvaluator()), files, "run", str(files / "bad.yaml"), str(files / "items.jsonl")
    )
    assert code == 2
    assert "bad.yaml is invalid:\n  questions: Dictionary should have at least 1 item" in err


def test_sample_prints_a_table_of_n_items(files: Path) -> None:
    fake = FakeEvaluator()
    code, out, _ = _invoke(
        _Factory(fake),
        files,
        "sample",
        str(files / "q.yaml"),
        str(files / "items.jsonl"),
        "-n",
        "2",
    )
    assert code == 0
    assert len(fake.calls) == 2
    assert "topic" in out
    assert "tech" in out


def test_calibrate_prints_the_recommended_threshold(files: Path) -> None:
    out = files / "out.jsonl"
    _invoke(
        _Factory(FakeEvaluator()),
        files,
        "run",
        str(files / "q.yaml"),
        str(files / "items.jsonl"),
        "-o",
        str(out),
    )
    labels = files / "labels.jsonl"
    labels.write_text(
        "\n".join(json.dumps({"id": i, "ad": i < 2}) for i in range(4)) + "\n", encoding="utf-8"
    )
    code, text, _ = _invoke(
        _Factory(FakeEvaluator()), files, "calibrate", str(out), str(labels), "-q", "ad"
    )
    assert code == 0
    assert "question ad: 4 labelled answers" in text
    assert "recommended: ad.noul >= " in text


def test_ask_prints_one_row(files: Path) -> None:
    fake = FakeEvaluator()
    code, out, _ = _invoke(_Factory(fake), files, "ask", "hello there", "-q", str(files / "q.yaml"))
    assert code == 0
    assert json.loads(out)["ad.noul"] == 0.9
    assert fake.calls[0][0] == "hello there"
