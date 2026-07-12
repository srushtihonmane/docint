"""Shared pytest fixtures — kept dependency-light (numpy + PyYAML only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def default_cfg() -> dict[str, Any]:
    """The parsed configs/default.yaml (the real file, not a copy)."""
    from docint.config import load_config

    return load_config()


@pytest.fixture()
def white_page() -> np.ndarray:
    """A clean white 300x400 BGR 'page' with one black text-like bar."""
    img = np.full((400, 300, 3), 255, dtype=np.uint8)
    img[60:80, 40:260] = 0
    return img
