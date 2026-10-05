"""`jev`: run typed TypeSafe Jev questions over a JSONL of items, sample them, calibrate them."""

import asyncio
import json
import logging
import os
import random
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractAsyncContextManager, contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Annotated

import typer
from pydantic import JsonValue
from rich.console import Console

from jev_kit import __version__
from jev_kit.cache import AnswerCache
from jev_kit.calibrate import calibrate
from jev_kit.client import open_evaluator, resolve_api_key
from jev_kit.errors import ItemsFailedError, JevError
from jev_kit.items import Item, read_items
from jev_kit.plan import RunPlan, build_plan, cost_usd, estimate_tokens, request_body
from jev_kit.render import (
    Progress,
    Summary,
    calibration_text,
    sample_table,
    summarize,
    to_row,
    write_rows,
)
from jev_kit.runner import Evaluation, Evaluator, ItemResult, Limits, execute
from jev_kit.spec import Question, Spec, load_spec

__all__ = ["EvaluatorFactory", "build_app", "main"]

type EvaluatorFactory = Callable[[], AbstractAsyncContextManager[Evaluator]]

CACHE_ENV = "JEV_KIT_CACHE"
# Defaults well under the published 80 req/s (https://docs.typesafe.ai/models.md): the limits
# move under load, and 429s cost retries. Raise --rps for big batches if 429s stay rare.
DEFAULT_CONCURRENCY = 16
DEFAULT_RPS = 20.0

QuestionsArg = Annotated[Path, typer.Argument(help="questions.yaml: model, state, questions.")]
ItemsArg = Annotated[Path, typer.Argument(help="Items JSONL: one object per line with an id.")]
IdFieldOpt = Annotated[str, typer.Option("--id-field", help="Item field holding the unique id.")]
ConcurrencyOpt = Annotated[
    int, typer.Option("--concurrency", min=1, help="Requests in flight at once.")
]
RpsOpt = Annotated[float, typer.Option("--rps", min=0.1, help="Requests started per second.")]
CacheOpt = Annotated[
    Path | None,
    typer.Option("--cache", help=f"Answer cache file (default ${CACHE_ENV} or ~/.cache/jev-kit)."),
]
NoCacheOpt = Annotated[
    bool, typer.Option("--no-cache", help="Neither read nor write the answer cache.")
]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="SDK request log on stderr.")]


@dataclass(frozen=True, slots=True)
class _RunOptions:
    limits: Limits
    cache_path: Path | None  # None: no cache


def build_app(make_evaluator: EvaluatorFactory) -> typer.Typer:
    app = typer.Typer(
        no_args_is_help=True,
        add_completion=False,
        help="Batch-evaluate texts with TypeSafe Jev: typed questions in, probabilities out.",
    )

    @app.callback(invoke_without_command=True)
    def _root(
        version: Annotated[bool, typer.Option("--version", help="Print the version.")] = False,
    ) -> None:
        if version:
            typer.echo(__version__)
            raise typer.Exit

    app.command()(_run_command(make_evaluator))
    app.command()(_sample_command(make_evaluator))
    app.command("calibrate")(calibrate_command)
    app.command()(_ask_command(make_evaluator))
    return app


def _run_command(make_evaluator: EvaluatorFactory) -> Callable[..., None]:
    def run(
        questions: QuestionsArg,
        items: ItemsArg,
        output: Annotated[
            Path | None, typer.Option("--output", "-o", help="Output JSONL (default stdout).")
        ] = None,
        limit: Annotated[int | None, typer.Option("--limit", min=1, help="First N items.")] = None,
        dry_run: Annotated[
            bool,
            typer.Option("--dry-run", help="Print the plan: first payload, requests, est. cost."),
        ] = False,
        id_field: IdFieldOpt = "id",
        concurrency: ConcurrencyOpt = DEFAULT_CONCURRENCY,
        rps: RpsOpt = DEFAULT_RPS,
        cache: CacheOpt = None,
        no_cache: NoCacheOpt = False,
        verbose: VerboseOpt = False,
    ) -> None:
        """Ask every question about every item; one request per item, answers cached."""
        with _errors():
            _logging(verbose=verbose)
            spec = load_spec(questions)
            loaded = read_items(items, id_field=id_field)[:limit]
            options = _options(concurrency, rps, cache, no_cache=no_cache)
            if dry_run:
                plan = _plan(spec, loaded, options)
                typer.echo(json.dumps(_dry_run(plan), ensure_ascii=False, indent=2))
                return
            results, summary = _evaluate(spec, loaded, options, make_evaluator)
            write_rows((to_row(r) for r in results), output)
            _note(summary.line())
            if summary.failed:
                raise ItemsFailedError(
                    f"{summary.failed} items failed (see `error` in the output); "
                    "rerun the same command to retry only them"
                )

    return run


def _sample_command(make_evaluator: EvaluatorFactory) -> Callable[..., None]:
    def sample(
        questions: QuestionsArg,
        items: ItemsArg,
        n: Annotated[int, typer.Option("-n", min=1, help="How many random items.")] = 20,
        seed: Annotated[
            int, typer.Option("--seed", help="Same seed, same items: compare wordings on them.")
        ] = 0,
        as_json: Annotated[
            bool, typer.Option("--json", help="Print the output rows as JSONL instead.")
        ] = False,
        id_field: IdFieldOpt = "id",
        concurrency: ConcurrencyOpt = DEFAULT_CONCURRENCY,
        rps: RpsOpt = DEFAULT_RPS,
        cache: CacheOpt = None,
        no_cache: NoCacheOpt = False,
        verbose: VerboseOpt = False,
    ) -> None:
        """Run on N random items and print a table to iterate on the wording."""
        with _errors():
            _logging(verbose=verbose)
            spec = load_spec(questions)
            loaded = read_items(items, id_field=id_field)
            picked = random.Random(seed).sample(  # noqa: S311 - picks items to look at, not secrets
                loaded, min(n, len(loaded))
            )
            options = _options(concurrency, rps, cache, no_cache=no_cache)
            results, summary = _evaluate(spec, picked, options, make_evaluator)
            if as_json:
                write_rows((to_row(r) for r in results), None)
            else:
                sample_table(results, list(spec.questions), Console(soft_wrap=False))
            _note(summary.line())

    return sample


def _ask_command(make_evaluator: EvaluatorFactory) -> Callable[..., None]:
    def ask(
        text: Annotated[str, typer.Argument(help="The text to evaluate.")],
        questions: Annotated[Path, typer.Option("--questions", "-q", help="questions.yaml.")],
        field: Annotated[
            str, typer.Option("--field", help="Item field the text goes into.")
        ] = "text",
        cache: CacheOpt = None,
        no_cache: NoCacheOpt = False,
        verbose: VerboseOpt = False,
    ) -> None:
        """Evaluate one text and print its output row."""
        with _errors():
            _logging(verbose=verbose)
            spec = load_spec(questions)
            item = Item(id="ask", fields={"id": "ask", field: text})
            options = _options(1, DEFAULT_RPS, cache, no_cache=no_cache)
            results, summary = _evaluate(spec, [item], options, make_evaluator)
            typer.echo(json.dumps(to_row(results[0]), ensure_ascii=False, indent=2))
            _note(summary.line())
            if summary.failed:
                raise ItemsFailedError(results[0].error or "the item failed")

    return ask


def calibrate_command(
    output: Annotated[Path, typer.Argument(help="The JSONL that `jev run` wrote.")],
    labels: Annotated[
        Path, typer.Argument(help='Labels JSONL: {"id": …, "<question>" or "expected": …}.')
    ],
    question: Annotated[str, typer.Option("--question", "-q", help="Question id.")],
    target: Annotated[
        float,
        typer.Option("--target", min=0, max=1, help="Accuracy a confidence gate must reach."),
    ] = 0.9,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Measure answers against hand labels and recommend a threshold."""
    with _errors():
        report, skipped = calibrate(
            _read_jsonl(output), _read_jsonl(labels), question, target=target
        )
        if as_json:
            typer.echo(json.dumps(_jsonable(report, skipped), ensure_ascii=False, indent=2))
        else:
            typer.echo(calibration_text(report, skipped))


def _options(concurrency: int, rps: float, cache: Path | None, *, no_cache: bool) -> _RunOptions:
    if no_cache:
        return _RunOptions(Limits(concurrency, rps), None)
    if cache is None:
        env = os.environ.get(CACHE_ENV)
        base = Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser()
        cache = Path(env).expanduser() if env else base / "jev-kit" / "answers.sqlite"
    return _RunOptions(Limits(concurrency, rps), cache)


def _plan(spec: Spec, items: list[Item], options: _RunOptions) -> RunPlan:
    if options.cache_path is None:
        return build_plan(spec, items, None)
    store = AnswerCache(options.cache_path)
    try:
        return build_plan(spec, items, store)
    finally:
        store.close()


def _dry_run(plan: RunPlan) -> dict[str, JsonValue]:
    bodies = [request_body(plan.spec, item) for item in plan.items if item.to_ask]
    tokens = [estimate_tokens(body) for body in bodies]
    return {
        "items": len(plan.items),
        "invalid_items": {str(i.id): i.error for i in plan.items if i.error},
        "requests": plan.requests,
        "cached_answers": sum(len(i.cached) for i in plan.items),
        "estimated_input_tokens": sum(tokens),
        "estimated_cost_usd": round(cost_usd(sum(tokens)), 6),
        "largest_request_tokens": max(tokens, default=0),
        "first_request": bodies[0] if bodies else None,
    }


def _evaluate(
    spec: Spec, items: list[Item], options: _RunOptions, make_evaluator: EvaluatorFactory
) -> tuple[list[ItemResult], Summary]:
    store = AnswerCache(options.cache_path) if options.cache_path else None
    started = time.monotonic()
    try:
        plan = build_plan(spec, items, store)

        async def go() -> list[ItemResult]:
            if not plan.requests:
                return await execute(plan, _NoRequests(), store, options.limits)
            _note(f"{len(plan.items)} items, {plan.requests} need a request")
            progress = Progress(len(plan.items))
            async with make_evaluator() as evaluator:
                return await execute(plan, evaluator, store, options.limits, progress)

        results = asyncio.run(go())
    finally:
        if store is not None:
            store.close()
    return results, summarize(results, time.monotonic() - started)


class _NoRequests:
    """Every answer is cached: no key needed, and a request here would be a bug."""

    async def evaluate(
        self, *, model: str, state: JsonValue, questions: Mapping[str, Question]
    ) -> Evaluation:
        raise AssertionError(f"a fully cached plan asked {model} {sorted(questions)} of {state!r}")


def _read_jsonl(path: Path) -> list[dict[str, JsonValue]]:
    return [item.fields for item in read_items(path)]


def _jsonable(report: object, skipped: dict[str, int]) -> dict[str, JsonValue]:
    if not is_dataclass(report) or isinstance(report, type):
        raise TypeError(report)
    body: dict[str, JsonValue] = asdict(report)
    body["not_compared"] = dict(skipped)
    return body


def _logging(*, verbose: bool) -> None:
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s", stream=sys.stderr)
        logging.getLogger("typesafe_sdk").setLevel(logging.INFO)


def _note(message: str) -> None:
    typer.echo(message, err=True)


@contextmanager
def _errors() -> Iterator[None]:
    try:
        yield
    except JevError as exc:
        _note(f"error: {exc}")
        raise typer.Exit(exc.exit_code) from None
    except KeyboardInterrupt:
        _note("interrupted; answers received so far are cached, rerun to continue")
        raise typer.Exit(130) from None


def main() -> None:
    def make_evaluator() -> AbstractAsyncContextManager[Evaluator]:
        return open_evaluator(resolve_api_key(os.environ))

    build_app(make_evaluator)()
