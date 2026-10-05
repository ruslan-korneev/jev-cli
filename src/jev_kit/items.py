"""Items JSONL: one JSON object per line, each with a unique id.

The id is how output rows, the cache report, labels and resumes refer to an item, so a missing
or repeated id is refused before the first request.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from jev_kit.errors import InputError

__all__ = ["Item", "ItemId", "read_items"]

type ItemId = str | int


@dataclass(frozen=True, slots=True)
class Item:
    id: ItemId
    fields: dict[str, JsonValue]


def read_items(path: Path, *, id_field: str = "id") -> list[Item]:
    items: list[Item] = []
    seen: dict[ItemId, int] = {}
    for number, row in _json_lines(path):
        if not isinstance(row, dict):
            raise InputError(f"{path}:{number}: expected a JSON object, got {type(row).__name__}")
        item_id = row.get(id_field)
        if isinstance(item_id, bool) or not isinstance(item_id, str | int):
            raise InputError(
                f"{path}:{number}: no string or integer {id_field!r} "
                f"(fields: {', '.join(sorted(row))}); pick the id field with --id-field"
            )
        if item_id in seen:
            raise InputError(f"{path}:{number}: id {item_id!r} repeats line {seen[item_id]}")
        seen[item_id] = number
        items.append(Item(id=item_id, fields=row))
    return items


def _json_lines(path: Path) -> list[tuple[int, JsonValue]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc.strerror}") from exc
    rows: list[tuple[int, JsonValue]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append((number, json.loads(line)))
        except json.JSONDecodeError as exc:
            raise InputError(f"{path}:{number}: not JSON ({exc.msg})") from exc
    return rows
