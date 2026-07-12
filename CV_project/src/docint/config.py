"""Configuration loading.

All tunable thresholds and parameters live in ``configs/default.yaml`` —
nothing is hardcoded in the pipeline modules. Modules receive plain nested
dicts (the sub-section named after them), which keeps them decoupled from
file I/O and trivially testable with literal dicts.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

# Repo root — this file sits at src/docint/config.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Default config shipped with the repo.
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "default.yaml"


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a YAML config file, optionally deep-merging ``overrides`` on top.

    Args:
        path: YAML file to load. Defaults to :data:`DEFAULT_CONFIG_PATH`.
        overrides: Nested mapping merged over the file contents (e.g. from
            CLI flags or per-request options). Dicts merge recursively;
            scalars and lists replace.

    Returns:
        The merged configuration as a plain nested ``dict``.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict with ``override`` recursively merged onto ``base``."""
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged
