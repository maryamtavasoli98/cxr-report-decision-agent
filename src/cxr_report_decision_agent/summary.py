"""Aggregation of the metrics JSON files into the numbers shown in the README and report."""

from __future__ import annotations

import numpy as np

from .evaluate import mean_std

METHODS = ["keyword_anywhere", "tools_fixed_rule", "agent", "tfidf"]
LABEL = {
    "keyword_anywhere": "B0 keyword anywhere (no negation)",
    "tools_fixed_rule": "B1 tools + fixed rule (no learning)",
    "agent": "B2 decision agent (tools + learned head)",
    "tfidf": "B3 TF-IDF logistic regression (no tools)",
}


def macro_f1(cls: dict, method: str) -> tuple[float, float]:
    return mean_std([np.mean([s["findings"][f][method]["f1"] for f in cls["findings"]]) for s in cls["seeds"]])


def per_finding(cls: dict, finding: str, method: str, key: str = "f1") -> tuple[float, float]:
    return mean_std([s["findings"][finding][method][key] for s in cls["seeds"]])


def macro_key(cls: dict, method: str, key: str) -> tuple[float, float]:
    return mean_std([np.mean([s["findings"][f][method][key] for f in cls["findings"]]) for s in cls["seeds"]])


def operating(cls: dict, method: str, key: str) -> tuple[float, float]:
    return mean_std([s["selective"][method]["operating_point"][key] for s in cls["seeds"]])


def curve(cls: dict, method: str, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cov = np.array([r["coverage"] for r in cls["seeds"][0]["selective"][method]["curve"]])
    vals = np.array([[r[key] for r in s["selective"][method]["curve"]] for s in cls["seeds"]])
    return cov, vals.mean(0), vals.std(0, ddof=1)


def audit_groups(audit: dict) -> dict:
    """Flip rates when the cited evidence sentence (vs a random other sentence of the same section) is deleted."""
    out = {}
    for name, sel in (
        ("present_correct", lambda r: r["decision"] == "present" and r["correct"]),
        ("present_wrong", lambda r: r["decision"] == "present" and not r["correct"]),
    ):
        rows = [r for r in audit["rows"] if sel(r)]
        ctrl = [r for r in rows if "flip_random" in r]
        out[name] = {
            "n": len(rows),
            "flip_cited": float(np.mean([r["flip_cited"] for r in rows])) if rows else float("nan"),
            "n_control": len(ctrl),
            "flip_random": float(np.mean([r["flip_random"] for r in ctrl])) if ctrl else float("nan"),
        }
    absent_wrong = [r for r in audit["rows"] if r["decision"] == "absent" and not r["correct"]]
    out["absent_wrong"] = {"n": len(absent_wrong), "by_context": {}}
    for r in absent_wrong:
        out["absent_wrong"]["by_context"][r["evidence_context"]] = (
            out["absent_wrong"]["by_context"].get(r["evidence_context"], 0) + 1
        )
    out["absent_correct_n"] = sum(1 for r in audit["rows"] if r["decision"] == "absent" and r["correct"])
    return out
