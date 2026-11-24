#!/usr/bin/env python3
"""Download the Open-i report XML (1.1 MB), parse it, write data/reports.jsonl and dataset.json."""

from cxr_report_decision_agent.common import environment, write_json
from cxr_report_decision_agent.data import build, labels

recs = build()
prev = {}
for r in recs:
    for k, v in labels(r).items():
        prev[k] = prev.get(k, 0) + v
write_json(
    "dataset.json",
    {
        "source": "Open-i NLMCXR_reports.tgz (Indiana University Chest X-ray Collection, CC BY-NC-ND 4.0)",
        "n_reports": len(recs),
        "n_with_impression": sum(bool(r["impression"]) for r in recs),
        "n_with_findings": sum(bool(r["findings"]) for r in recs),
        "positives_per_finding": prev,
        "environment": environment(),
    },
)
print(len(recs), prev)
