"""Items JSONL parsing: every item needs a unique string or integer id.

Pins the refusal messages (line number + what to do). Does not cover large-file performance.
"""

from pathlib import Path

import pytest

from jev_kit.errors import InputError
from jev_kit.items import read_items


def _write(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "items.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_items_keep_order_and_skip_blank_lines(tmp_path: Path) -> None:
    items = read_items(_write(tmp_path, '{"id": 2, "text": "b"}', "", '{"id": "a", "text": "x"}'))
    assert [item.id for item in items] == [2, "a"]
    assert items[0].fields == {"id": 2, "text": "b"}


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (['{"text": "no id"}'], r"items.jsonl:1: no string or integer 'id' \(fields: text\)"),
        (['{"id": true}'], "no string or integer 'id'"),
        (['{"id": 1.5}'], "no string or integer 'id'"),
        (['{"id": 1}', '{"id": 1}'], "items.jsonl:2: id 1 repeats line 1"),
        (["[1, 2]"], "expected a JSON object, got list"),
        (["{not json"], "items.jsonl:1: not JSON"),
    ],
)
def test_bad_items_are_refused_with_the_line(
    tmp_path: Path, lines: list[str], message: str
) -> None:
    with pytest.raises(InputError, match=message):
        read_items(_write(tmp_path, *lines))


def test_id_field_can_be_chosen(tmp_path: Path) -> None:
    items = read_items(_write(tmp_path, '{"msg_id": 7, "text": "a"}'), id_field="msg_id")
    assert items[0].id == 7


def test_missing_file_is_an_input_error(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="cannot read"):
        read_items(tmp_path / "nope.jsonl")
