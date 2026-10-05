"""questions.yaml: the model, how an item becomes `state`, and the typed questions.

The question shapes mirror https://docs.typesafe.ai/api.md and are validated here, before any
request, so a typo in the file fails locally instead of once per item.
"""

import re
import string
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator

from jev_kit.errors import InputError

__all__ = [
    "ChoiceQuestion",
    "Content",
    "NoulCriteria",
    "NoulQuestion",
    "Question",
    "ScoreQuestion",
    "Spec",
    "build_state",
    "load_spec",
]

DEFAULT_MODEL = "jev-latest"
# Limits from https://docs.typesafe.ai/api.md: "a maximum of 255 options per Choice"; "A Score
# should have at least two levels; the API accepts up to 10."
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

# A question id becomes the prefix of flat output keys (`<id>.noul`), so it must not contain a dot.
_QUESTION_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")

# Instructions, option descriptions and level descriptions: text, a JSON object, or an array.
type Content = str | dict[str, JsonValue] | list[JsonValue]

_CLOSED = ConfigDict(frozen=True, extra="forbid")


class NoulCriteria(BaseModel):
    model_config = _CLOSED

    true: Content | None = None
    false: Content | None = None


class NoulQuestion(BaseModel):
    model_config = _CLOSED

    type: Literal["noul"]
    instructions: Content
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    model_config = _CLOSED

    type: Literal["choice"]
    instructions: Content
    criteria: dict[str, Content | None]

    @field_validator("criteria")
    @classmethod
    def _option_count(cls, value: dict[str, Content | None]) -> dict[str, Content | None]:
        if not 2 <= len(value) <= MAX_CHOICE_OPTIONS:  # noqa: PLR2004 - a choice needs two options
            raise ValueError(f"a choice needs 2..{MAX_CHOICE_OPTIONS} options, got {len(value)}")
        return value


class ScoreQuestion(BaseModel):
    model_config = _CLOSED

    type: Literal["score"]
    instructions: Content
    criteria: list[Content]

    @field_validator("criteria")
    @classmethod
    def _level_count(cls, value: list[Content]) -> list[Content]:
        if not MIN_SCORE_LEVELS <= len(value) <= MAX_SCORE_LEVELS:
            raise ValueError(
                f"a score needs {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} levels, got {len(value)}"
            )
        return value


type Question = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")
]


class Spec(BaseModel):
    model_config = _CLOSED

    model: str = DEFAULT_MODEL
    # A format template over item fields ("{text}") → a string state, or a list of field names →
    # an object state holding just those fields.
    state: str | list[str]
    questions: dict[str, Question] = Field(min_length=1)

    @field_validator("state")
    @classmethod
    def _state_fields(cls, value: str | list[str]) -> str | list[str]:
        fields = value if isinstance(value, list) else template_fields(value)
        if not fields:
            raise ValueError('state names no item field; use a template like "{text}" or a list')
        bad = [name for name in fields if not _FIELD_NAME.match(name)]
        if bad:
            raise ValueError(f"state field names must be plain identifiers, got {bad}")
        return value

    @field_validator("questions")
    @classmethod
    def _question_ids(cls, value: dict[str, Question]) -> dict[str, Question]:
        bad = [qid for qid in value if not _QUESTION_ID.match(qid)]
        if bad:
            raise ValueError(f"question ids may use letters, digits, _ and - only, got {bad}")
        return value

    @property
    def state_fields(self) -> list[str]:
        return self.state if isinstance(self.state, list) else template_fields(self.state)


def template_fields(template: str) -> list[str]:
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(
            f"state template {template!r} is not a valid format string: {exc}"
        ) from exc
    return [name for _, name, _, _ in parsed if name is not None]


def build_state(spec: Spec, item: dict[str, JsonValue]) -> JsonValue:
    """The `state` sent for one item. Raises InputError naming the missing fields."""
    missing = [name for name in spec.state_fields if item.get(name) is None]
    if missing:
        raise InputError(f"item has no {', '.join(missing)} (fields: {', '.join(sorted(item))})")
    if isinstance(spec.state, list):
        return {name: item[name] for name in spec.state}
    return spec.state.format_map(item)


class _Loader(yaml.SafeLoader):
    """YAML with JSON's notion of keys: every mapping key is a string.

    PyYAML reads `true:`, `yes:` and `no:` keys as booleans and `1:` as an int. Here a choice
    option named `yes` must stay "yes", and a noul criteria key `true` must become "true".
    """


def _string_keys(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict[str, object]:
    pairs = loader.construct_pairs(node, deep=True)
    return {
        (str(key).lower() if isinstance(key, bool) else str(key)): value for key, value in pairs
    }


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _string_keys)
# YAML 1.2 booleans only: `yes`/`no`/`on`/`off` stay strings anywhere in the file.
_Loader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)


def load_spec(path: Path) -> Spec:
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader)  # noqa: S506 - _Loader is a SafeLoader
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc.strerror}") from exc
    except yaml.YAMLError as exc:
        raise InputError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise InputError(f"{path} must be a mapping with model, state and questions")
    try:
        return Spec.model_validate(raw)
    except ValidationError as exc:
        raise InputError(f"{path} is invalid:\n{_format_errors(exc)}") from exc


def _format_errors(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors(include_url=False):
        location = ".".join(str(part) for part in error["loc"])
        lines.append(f"  {location}: {error['msg']}")
    return "\n".join(lines)
