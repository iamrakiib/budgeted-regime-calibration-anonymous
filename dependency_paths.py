"""Repository-local paths for non-vendored public research-code dependencies."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

THIRD_PARTY_ROOT = Path(
    os.environ.get(
        "BRC_THIRD_PARTY_ROOT",
        REPO_ROOT / "third_party",
    )
).expanduser().resolve()


def third_party_path(name: str) -> Path:
    """Return the expected location of a public third-party source tree."""
    return THIRD_PARTY_ROOT / name