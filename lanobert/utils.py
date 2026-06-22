"""Shared helpers: config loading, seeding, device, simple namespace access."""
from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import yaml


class Config(dict):
    """dict that also supports attribute access and nested dotted lookups."""

    __getattr__ = dict.get

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if isinstance(v, dict):
                self[k] = Config(v)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str) -> Config:
    """Load a YAML config file into a Config object."""
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)
    cfg = Config(raw)
    cfg._source_path = os.path.abspath(path)
    return cfg


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch (if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """Return the best available torch device, or 'cpu' if torch is missing."""
    try:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        return "cpu"


def ensure_dir(path: str) -> str:
    """Create a directory (and parents) if it does not exist; return the path."""
    os.makedirs(path, exist_ok=True)
    return path
