# jev-kit

`jev`: a CLI that runs typed TypeSafe Jev questions (noul / choice / score) over a JSONL of items —
one request per item with every question, answers cached per question — plus `sample` to tune
wording and `calibrate` to pick thresholds from hand labels. The agent skill is `skills/jev/`
(Agent Skills format). The installed skill is a copy (`~/.agents/skills/jev`, which
`~/.claude/skills/jev` links to), so after editing `skills/jev/` reinstall it:
`npx skills add . --skill jev -g -y`. Keep it agent-neutral and in step with the CLI.

## Commands

```bash
uv run pytest                              # unit (no network)
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy                                # strict
uv tool install --editable . --reinstall   # refresh the `jev` on PATH
```

## Where things are

- **Source of truth for the API:** <https://docs.typesafe.ai/llms.txt> (pages end in `.md`).
  Check limits, prices and SDK behaviour there, not from memory.
- **Contracts:** the output row format is in `answers.py` and README "Output"; exit codes in
  `errors.py`; the questions.yaml schema in `spec.py`. Changing any of them is a contract change:
  update the README and the skill in the same change.
- **TypeSafe boundary:** `client.py` is the only importer of `typesafe_sdk` (official SDK; retries
  live in its `RetryPolicy`). The runner depends on the `Evaluator` protocol in `runner.py`.
- **Plan → execute:** `plan.py` builds the per-item plan (state, cache hits, questions to ask);
  `--dry-run` prints it and `runner.py` executes the same object. No request path bypasses the plan.
- **State:** the answer cache (`cache.py`, SQLite at `$JEV_KIT_CACHE` or `~/.cache/jev-kit/`) is
  written only by the runner, only with complete answers.

## Rules for this repo

- Releases: the version in `pyproject.toml` and `jev_kit/__init__.py` equals the git tag
  (`v0.1.0`), and the install lines in `skills/jev/SKILL.md` and README pin that tag
  (`…/jev-cli@vX.Y.Z`). Bump all of them together, then push the commit and the tag;
  `tests/unit/test_release_pins.py` fails on a mismatch.
- The API key comes from `TYPESAFE_API_KEY` or `~/.config/typesafe/api_key`. Never print, log or
  commit it; tests use a fake key and a mock transport.
- Recorded API responses live in `tests/fixtures/typesafe/`; re-record them rather than hand-edit.
- No real channel names, ids or personal data in code, tests or docs; use `@example_channel`.
