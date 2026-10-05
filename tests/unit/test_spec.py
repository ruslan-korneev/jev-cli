"""questions.yaml validation and state building.

Pins: the question limits from the API reference, YAML key handling (`true:`/`yes:` keys), and
how an item becomes `state`. Does not cover how the API itself reacts to a valid spec.
"""

from pathlib import Path

import pytest
from pydantic import JsonValue

from jev_kit.errors import InputError
from jev_kit.spec import Spec, build_state, load_spec

NOUL = "  q:\n    type: noul\n    instructions: Is it an ad?\n"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "questions.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("state: '{text}'\nquestions: {}\n", "at least 1 item"),
        ("questions:\n" + NOUL, "state"),
        ("state: 'no fields'\nquestions:\n" + NOUL, "names no item field"),
        ("state: '{a.b}'\nquestions:\n" + NOUL, "plain identifiers"),
        ("state: '{text}'\nquestions:\n  a.b:\n    type: noul\n    instructions: x\n", "ids may"),
        ("state: '{text}'\nquestions:\n  q:\n    type: guess\n    instructions: x\n", "guess"),
        (
            (
                "state: '{text}'\nquestions:\n  q:\n    type: choice\n    instructions: x\n"
                "    criteria: {only: null}\n"
            ),
            "2..255 options",
        ),
        (
            (
                "state: '{text}'\nquestions:\n  q:\n    type: score\n    instructions: x\n"
                "    criteria: [a, b, c, d, e, f, g, h, i, j, k]\n"
            ),
            "2..10 levels",
        ),
        ("state: '{text}'\nquestions:\n" + NOUL + "    extra: 1\n", "extra"),
        ("- just\n- a list\n", "must be a mapping"),
        ("state: [\n", "not valid YAML"),
    ],
)
def test_invalid_spec_is_refused_with_the_reason(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(InputError, match=message):
        load_spec(_write(tmp_path, body))


def test_yaml_boolean_looking_keys_stay_strings(tmp_path: Path) -> None:
    spec = load_spec(
        _write(
            tmp_path,
            "state: '{text}'\nquestions:\n"
            "  ad:\n    type: noul\n    instructions: Is it an ad?\n"
            "    criteria:\n      true: Sells something\n      false: Sells nothing\n"
            "  answer:\n    type: choice\n    instructions: Does it agree?\n"
            "    criteria:\n      yes: Agrees\n      no: Disagrees\n      1: A number option\n",
        )
    )
    dumped = {qid: q.model_dump(exclude_none=True) for qid, q in spec.questions.items()}
    assert dumped["ad"]["criteria"] == {"true": "Sells something", "false": "Sells nothing"}
    assert list(dumped["answer"]["criteria"]) == ["yes", "no", "1"]


def test_model_defaults_to_jev_latest(tmp_path: Path) -> None:
    assert load_spec(_write(tmp_path, "state: '{text}'\nquestions:\n" + NOUL)).model == "jev-latest"


@pytest.mark.parametrize(
    ("state", "item", "expected"),
    [
        ("{text}", {"id": 1, "text": "hello"}, "hello"),
        ("Title: {title}\n\n{text}", {"title": "T", "text": "body"}, "Title: T\n\nbody"),
        (["text", "views"], {"text": "hi", "views": 5, "id": 1}, {"text": "hi", "views": 5}),
    ],
)
def test_state_is_built_from_item_fields(
    state: str | list[str], item: dict[str, JsonValue], expected: JsonValue
) -> None:
    spec = Spec.model_validate(
        {"state": state, "questions": {"q": {"type": "noul", "instructions": "x"}}}
    )
    assert build_state(spec, item) == expected


@pytest.mark.parametrize("item", [{"id": 1}, {"id": 1, "text": None}])
def test_missing_state_field_names_the_field_and_what_exists(item: dict[str, JsonValue]) -> None:
    spec = Spec.model_validate(
        {"state": "{text}", "questions": {"q": {"type": "noul", "instructions": "x"}}}
    )
    with pytest.raises(InputError, match=r"no text \(fields: id"):
        build_state(spec, item)
