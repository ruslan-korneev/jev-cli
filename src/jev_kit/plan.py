"""The plan of a run: per item, its state, which answers the cache holds, and what is left to ask.

`jev run --dry-run` prints this plan and the real run executes the same object, so the dry run's
request count and first payload are the ones the run will send.
"""

import json
from dataclasses import dataclass, field

from pydantic import JsonValue

from jev_kit.cache import AnswerCache, CachedAnswer, question_key
from jev_kit.errors import InputError, JevError
from jev_kit.items import Item, ItemId
from jev_kit.spec import Spec, build_state

__all__ = [
    "PRICE_PER_MTOK_USD",
    "ItemPlan",
    "RunPlan",
    "build_plan",
    "cost_usd",
    "estimate_tokens",
    "request_body",
]

# Input tokens only; output tokens are free. https://docs.typesafe.ai/models.md, jev-1.13.0,
# read 2026-10-03. A different model may be priced differently.
PRICE_PER_MTOK_USD = 0.042

# Rough input-token estimate for --dry-run, fitted by least squares on usage.input_tokens of 50
# live requests (Russian Telegram posts, the English example questions; 2026-10-03, jev-1.13.0):
# 330 + ASCII chars / 5.3 + other chars / 1.34. Total was within 0.01%, single items within 8%.
# Cyrillic costs ~4x more per character than English, hence the split.
_TOKENS_OVERHEAD = 330
_ASCII_CHARS_PER_TOKEN = 5.3
_OTHER_CHARS_PER_TOKEN = 1.34


@dataclass(frozen=True, slots=True)
class ItemPlan:
    id: ItemId
    state: JsonValue
    keys: dict[str, str] = field(default_factory=dict)  # question id → cache key
    cached: dict[str, CachedAnswer] = field(default_factory=dict)
    to_ask: tuple[str, ...] = ()
    error: str | None = None  # the item cannot be sent (e.g. a state field is missing)


@dataclass(frozen=True, slots=True)
class RunPlan:
    spec: Spec
    items: tuple[ItemPlan, ...]

    @property
    def requests(self) -> int:
        return sum(1 for item in self.items if item.to_ask)


def build_plan(spec: Spec, items: list[Item], cache: AnswerCache | None) -> RunPlan:
    drafts: list[ItemPlan] = []
    for item in items:
        try:
            state = build_state(spec, item.fields)
        except JevError as exc:
            drafts.append(ItemPlan(id=item.id, state=None, error=str(exc)))
            continue
        keys = {qid: question_key(spec.model, state, q) for qid, q in spec.questions.items()}
        drafts.append(ItemPlan(id=item.id, state=state, keys=keys))

    if items and all(draft.error for draft in drafts):
        # Every item failing the same way is a template typo, not bad data.
        raise InputError(f"no item can build its state: {drafts[0].error}")

    hits = cache.get_many([k for d in drafts for k in d.keys.values()]) if cache else {}
    planned = []
    for draft in drafts:
        cached = {qid: hits[key] for qid, key in draft.keys.items() if key in hits}
        to_ask = tuple(qid for qid in draft.keys if qid not in cached)
        planned.append(
            ItemPlan(id=draft.id, state=draft.state, keys=draft.keys, cached=cached, to_ask=to_ask)
            if draft.error is None
            else draft
        )
    return RunPlan(spec=spec, items=tuple(planned))


def request_body(spec: Spec, item: ItemPlan) -> dict[str, JsonValue]:
    """The POST /v1/systemone body for this item's unanswered questions."""
    return {
        "state": item.state,
        "model": spec.model,
        "questions": {
            qid: spec.questions[qid].model_dump(mode="json", exclude_none=True)
            for qid in item.to_ask
        },
    }


def estimate_tokens(body: dict[str, JsonValue]) -> int:
    text = json.dumps(body, ensure_ascii=False)
    ascii_chars = sum(char.isascii() for char in text)
    other_chars = len(text) - ascii_chars
    return _TOKENS_OVERHEAD + round(
        ascii_chars / _ASCII_CHARS_PER_TOKEN + other_chars / _OTHER_CHARS_PER_TOKEN
    )


def cost_usd(input_tokens: int) -> float:
    return input_tokens * PRICE_PER_MTOK_USD / 1_000_000
