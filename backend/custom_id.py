"""The `custom:` model-id prefix — identity, not registration.

A custom model is stored under a local cache and addressed as
`custom:<name>`. Whether an id is one of those, and where its files
live, is a fact about the id. Registration, the CLI, and Hub scanning
are not. Hub metadata asks this module whether to read a local
`checkpoint_header.json` instead of the Hub; it does not import
`custom_model`.
"""
from __future__ import annotations

from pathlib import Path


NAME_PREFIX = "custom:"
CACHE_DIR = Path.home() / ".huggingarch" / "custom_models"


def cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def is_custom_id(model_id: str) -> bool:
    return isinstance(model_id, str) and model_id.startswith(NAME_PREFIX)


def strip_prefix(model_id: str) -> str:
    return model_id[len(NAME_PREFIX):] if is_custom_id(model_id) else model_id


def model_path(name: str) -> Path:
    return cache_dir() / strip_prefix(name)
