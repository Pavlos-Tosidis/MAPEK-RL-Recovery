"""Path handling and reproducible run metadata."""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def versions():
    packages = ("torch", "stable-baselines3", "gymnasium", "numpy", "opencv-python", "ultralytics")
    answer = {}
    for package in packages:
        try:
            answer[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            answer[package] = "not installed"
    return answer


def metadata(kind, **details):
    return dict(kind=kind, created_utc=datetime.now(timezone.utc).isoformat(),
                dependencies=versions(), **details)


def unique_directory(base: str | Path):
    """Disallow accidental overwrites; use a new run name."""
    p = Path(base).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=False)
    return p
