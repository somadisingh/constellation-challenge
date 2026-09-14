"""YAML configuration for Experiment 1 (plan §13)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / 'configs'
DEFAULT_CONFIG = CONFIG_DIR / 'default.yaml'


def load_config(path: str | Path | None = None) -> dict:
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if path:
        override = yaml.safe_load(Path(path).read_text()) or {}
        cfg = deep_merge(cfg, override)
    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out
