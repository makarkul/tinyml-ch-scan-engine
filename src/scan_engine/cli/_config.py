from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args(argv)
    with args.config.open() as f:
        return yaml.safe_load(f)
