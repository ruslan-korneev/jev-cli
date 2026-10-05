# jev-kit

`jev`: run typed TypeSafe Jev questions over hundreds of texts from the command line. Each item of
a JSONL file becomes one request carrying every question; answers come back as probabilities and
land in one flat JSONL row per item, cached on disk so reruns and resumed runs cost nothing.

Built for agents: the skill in [`skills/jev/`](skills/jev/SKILL.md) tells them when and how to use it.

The API itself is documented at <https://docs.typesafe.ai/llms.txt>; that is the source of truth
for question types, limits and prices.

## Install

```bash
uv tool install --editable .            # puts `jev` on PATH (~/.local/bin)
npx skills add . --skill jev -g -y      # installs the agent skill (~/.agents/skills, linked into ~/.claude/skills)
```

The API key comes from `TYPESAFE_API_KEY`, or from `~/.config/typesafe/api_key` when the variable
is unset. It is never printed.

## Quick start

```bash
tg read @example_channel -n 200 --full --json | jq -c '.[] | select(.text) | {id, text}' > posts.jsonl
jev run examples/questions.yaml posts.jsonl --dry-run      # first request body + estimated cost
jev sample examples/questions.yaml posts.jsonl -n 20       # table of 20 random items
jev run examples/questions.yaml posts.jsonl -o out.jsonl
jq -c 'select(.["is_ad.noul"] < 0.5) | {id, useful: .["usefulness.score"]}' out.jsonl
```

## questions.yaml

See [`examples/questions.yaml`](examples/questions.yaml).

| Key | Meaning |
| - | - |
| `model` | Model name or alias, default `jev-latest`. Pin a version (`jev-1.13.0`) once thresholds are tuned: an alias moves when a new model ships. |
| `state` | `"{text}"`-style template over item fields → a string state; or a list of field names → an object state with those fields. |
| `questions` | Map of id → `{type: noul\|choice\|score, instructions, criteria}`. `instructions` and the criteria descriptions may be strings, objects or lists ([advanced](https://docs.typesafe.ai/primitives/advanced.md)). Ids use letters, digits, `_`, `-`. |

The file is validated before any request: 2–255 choice options, 2–10 score levels, closed keys.
YAML keys are read as JSON keys, so `true:`/`false:` under a noul's `criteria` and choice options
named `yes`/`no` stay strings.

## Commands

| Command | What it does |
| - | - |
| `jev run Q ITEMS [-o OUT]` | Every question on every item. `--limit N`, `--dry-run`, `--concurrency 16`, `--rps 20`, `--id-field id`, `--cache PATH`, `--no-cache`, `-v`. |
| `jev sample Q ITEMS -n 20` | N random items (`--seed`, default 0, so the same items come back while you tune wording) as a table; `--json` for rows. |
| `jev calibrate OUT LABELS -q Q` | Labels are JSONL `{"id": …, "<Q>" or "expected": …}`. noul: accuracy/precision/recall/F1 per threshold 0.05..0.95, recommends the best F1. choice/score: accuracy (score: exact and ±1 on the rounded score, MAE) and the lowest confidence gate reaching `--target` accuracy. `--json` for the report. |
| `jev ask TEXT -q Q` | One text (as the item field `--field text`), prints its row. |

### Output

One JSON object per item, in input order:

```json
{"id": 4436, "model": "jev-1.13.0",
 "is_ad.noul": 0.97, "is_ad.confidence": 0.94,
 "topic.choice": "technology", "topic.confidence": 0.99,
 "topic.probabilities": {"culture": 0.0, "economy": 0.0, "technology": 1.0, "other": 0.0, "politics": 0.0},
 "usefulness.score": 2.1, "usefulness.confidence": 0.89,
 "usefulness.probabilities": {"0": 0.0, "1": 0.0, "2": 0.9, "3": 0.1},
 "usage": {"input_tokens": 1215, "output_tokens": 89}, "cached_answers": 0, "error": null}
```

- `<q>.confidence` is the API's value for choice and score. A noul has none in the API; the column
  holds |2p − 1|, the formula from [Confidence](https://docs.typesafe.ai/confidence.md#noul).
- `usage` is what this run paid for the item: zeros when every answer came from the cache.
- A failed item keeps its `id`, gets `error` and no answer keys; the run continues and exits 4.

Progress and the summary (`items, requests, answers from cache, input tokens, $cost, seconds`) go to
stderr.

### Exit codes

| Code | Meaning |
| - | - |
| 0 | Every item answered. |
| 1 | Unexpected failure. |
| 2 | Bad input: questions.yaml, items, labels or flags. The message says what and where. |
| 3 | No API key, or the key was rejected (the run stops at the first rejection). |
| 4 | The run finished, some rows carry `error`. Rerun the same command to retry just those. |

## Cache, resume, retries

- The cache is SQLite at `$JEV_KIT_CACHE`, else `$XDG_CACHE_HOME/jev-kit/answers.sqlite`
  (`~/.cache/jev-kit/answers.sqlite`). One row per hash of (requested model, exact state, exact
  question). Questions in one request are evaluated independently
  ([primitives](https://docs.typesafe.ai/primitives.md)), so editing one question re-asks only
  that question, and a rerun or a resumed run asks only what is missing. A request carries all of
  an item's missing questions at once.
- The key uses the requested model name: with `jev-latest`, answers cached before an alias move
  are reused. Pin the version, or pass `--no-cache`, when that matters.
- Only complete responses are cached. A failed item is retried on the next run.
- Retries are the official SDK's (`typesafe-sdk`): 408/429/5xx/529 and connection errors, honouring
  `retry-after`/`retry-after-ms`, otherwise exponential backoff; up to 6 retries within 180 s per
  item, 60 s per attempt. Requests are capped by `--concurrency` (in flight) and `--rps` (started
  per second); the defaults stay well under the published 80 req/s.

### Token estimate

`--dry-run` estimates input tokens as `330 + ASCII chars / 5.3 + other chars / 1.34` of each
request body (Cyrillic costs about four times more per character than English). The constants were
fitted on 50 live requests (Russian Telegram posts, English questions): the total matched within
0.01% and single items within 8% — on the data it was fitted on. Treat it as an estimate.

## Development

```bash
uv run pytest                                  # unit, no network
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy                                    # strict
uv tool install --editable . --reinstall       # refresh `jev` on PATH
```
