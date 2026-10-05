"""One base error per package; the exit code travels with the error class.

Exit codes (the CLI contract, also in README "Exit codes"):

- 0: done, every item answered;
- 1: unexpected upstream failure outside a per-item loop;
- 2: bad input (questions.yaml, items JSONL, labels, flags);
- 3: no API key, or the key was rejected;
- 4: the run finished but some items carry an `error` (see the output rows).
"""

__all__ = ["AuthError", "InputError", "ItemsFailedError", "JevError", "UpstreamError"]


class JevError(Exception):
    exit_code = 1


class InputError(JevError):
    exit_code = 2


class AuthError(JevError):
    """The key is missing or rejected. Fatal for a run: every further request would fail too."""

    exit_code = 3


class UpstreamError(JevError):
    """One request failed after the SDK's retries. Recorded on the item; the run goes on."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ItemsFailedError(JevError):
    exit_code = 4
