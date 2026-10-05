"""On-disk answer cache, one row per (model, state, question).

The key is a hash of the requested model name, the exact `state` and the exact question body, so
editing one question re-asks only that question, and a rerun over the same items asks nothing.
Questions in one request are evaluated independently ("You can add or remove questions without
changing the others' results", https://docs.typesafe.ai/primitives.md), which is what makes a
per-question cache sound even though requests carry every question of an item.

Only complete answers are written; a failed request writes nothing.
"""

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue, ValidationError

from jev_kit.answers import Answer
from jev_kit.spec import Question

__all__ = ["AnswerCache", "CachedAnswer", "question_key"]

# Bump when the key recipe or the stored Answer shape changes; old rows then simply miss.
_KEY_VERSION = 1


@dataclass(frozen=True, slots=True)
class CachedAnswer:
    model: str  # the versioned model that answered (e.g. jev-1.13.0), not the requested alias
    answer: Answer


def question_key(model: str, state: JsonValue, question: Question) -> str:
    body = {
        "v": _KEY_VERSION,
        "model": model,
        "state": state,
        "question": question.model_dump(mode="json", exclude_none=True),
    }
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class AnswerCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS answers ("
            " key TEXT PRIMARY KEY, model TEXT NOT NULL, answer TEXT NOT NULL,"
            " created_at REAL NOT NULL)"
        )

    def get_many(self, keys: Sequence[str]) -> dict[str, CachedAnswer]:
        found: dict[str, CachedAnswer] = {}
        corrupt: list[str] = []
        for key, model, raw in self._select(keys):
            try:
                found[key] = CachedAnswer(model=model, answer=Answer.model_validate_json(raw))
            except ValidationError:
                corrupt.append(key)  # undecodable or old shape: a miss, and gone
        if corrupt:
            with self._db:
                self._db.executemany("DELETE FROM answers WHERE key = ?", [(k,) for k in corrupt])
        return found

    def put_many(self, entries: Iterable[tuple[str, CachedAnswer]]) -> None:
        now = time.time()
        rows = [(key, c.model, c.answer.model_dump_json(), now) for key, c in entries]
        with self._db:
            self._db.executemany("INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?)", rows)

    def close(self) -> None:
        self._db.close()

    def _select(self, keys: Sequence[str]) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        # SQLite caps bound parameters per statement (999 on older builds).
        for start in range(0, len(keys), 500):
            chunk = keys[start : start + 500]
            marks = ",".join("?" * len(chunk))
            rows.extend(
                self._db.execute(
                    f"SELECT key, model, answer FROM answers WHERE key IN ({marks})",  # noqa: S608 - only "?" marks are interpolated
                    chunk,
                )
            )
        return rows
