"""Load a document type's configuration: its name and the fields to extract."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def field_names(config: dict) -> list[str]:
    return [field["name"] for field in config["fields"]]
