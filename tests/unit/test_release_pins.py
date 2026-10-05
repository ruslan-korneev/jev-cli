"""Pins the release contract: the install lines agents and people copy (SKILL.md,
README) point at the tag of the current version, and both version strings agree.

An unpinned or stale line installs whatever master happens to be (flagged by the
skills.sh supply-chain audit). Not covered: that the tag exists on GitHub.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from jev_kit import __version__

ROOT = Path(__file__).resolve().parents[2]
REPO = "github.com/ruslan-korneev/jev-cli"


def test_package_versions_agree() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["version"] == __version__


@pytest.mark.parametrize("doc", ["skills/jev/SKILL.md", "README.md"])
def test_every_git_install_line_pins_the_current_tag(doc: str) -> None:
    text = (ROOT / doc).read_text()
    refs = re.findall(rf"git\+https://{re.escape(REPO)}(@[^'\s`]+)?", text)
    assert refs, f"{doc} has no git install line"
    assert set(refs) == {f"@v{__version__}"}, f"{doc} install lines: {refs}"
