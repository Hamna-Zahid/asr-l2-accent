"""Configuration loading and lightweight access helpers.

Every CLI script loads ``config/default.yaml`` and may override individual
fields from command-line arguments. Keep this module dependency-light so it
imports fast in every entry point.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

# Repo root = three levels up from this file (src/asr_l2/config.py -> repo).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


def load_dotenv(path: str | Path | None = None) -> None:
    """Load simple KEY=VALUE lines from a .env file into os.environ.

    Minimal, dependency-free. Existing environment variables win (we never
    overwrite an already-set var). Lines starting with '#' are ignored.
    Looks at the repo-root .env by default. Secrets stay out of git via
    .gitignore; this just makes `HF_TOKEN` etc. available to the scripts.
    """
    env_path = Path(path) if path else REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config into a plain dict.

    Relative ``paths.*`` entries are resolved to absolute paths against the
    repo root so scripts work regardless of the current working directory.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    paths = cfg.setdefault("paths", {})
    for key, value in list(paths.items()):
        p = Path(value)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        paths[key] = str(p)
    return cfg


def deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out
