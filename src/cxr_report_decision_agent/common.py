"""Shared helpers: paths, config, environment record, JSON output."""

from __future__ import annotations

import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "data" / "reports.jsonl"
URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"


def config() -> dict:
    return json.loads((ROOT / "configs" / "default.json").read_text())


def environment() -> dict:
    import numpy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "machine": platform.machine(),
        "platform": platform.system(),
        "compute": "CPU only",
    }


def write_json(name: str, obj) -> None:
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / name).write_text(json.dumps(obj, indent=1))


def read_json(name: str):
    return json.loads((METRICS / name).read_text())
