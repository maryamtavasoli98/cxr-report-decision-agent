#!/usr/bin/env python3
"""Three figures from results/metrics."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cxr_report_decision_agent.common import FIGURES, read_json
from cxr_report_decision_agent.summary import LABEL, METHODS, audit_groups, curve, per_finding

FIGURES.mkdir(parents=True, exist_ok=True)
cls, audit = read_json("classification.json"), read_json("audit.json")
F = cls["findings"]
COL = {"keyword_anywhere": "#bbbbbb", "tools_fixed_rule": "#7fb3d5", "agent": "#c0392b", "tfidf": "#555555"}

fig, ax = plt.subplots(figsize=(9, 3.8))
w = 0.2
for k, m in enumerate(METHODS):
    mu = [per_finding(cls, f, m)[0] for f in F]
    sd = [per_finding(cls, f, m)[1] for f in F]
    ax.bar(np.arange(len(F)) + (k - 1.5) * w, mu, w, yerr=sd, label=LABEL[m], color=COL[m], capsize=1.5)
ax.set_xticks(range(len(F)), F, rotation=25, ha="right")
ax.set_ylabel("F1 on test reports (mean ± sd, 5 splits)")
ax.legend(fontsize=7, loc="upper right", ncol=2)
ax.set_ylim(0, 1.15)
fig.tight_layout()
fig.savefig(FIGURES / "f1_by_finding.png", dpi=160)

fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
for m in ("agent", "tfidf"):
    for ax, key, yl in (
        (axs[0], "positives_abstained", "fraction of true positives abstained on"),
        (axs[1], "error_rate", "error rate on answered pairs"),
    ):
        cov, mu, sd = curve(cls, m, key)
        ax.plot(cov, mu, color=COL[m], label=LABEL[m])
        ax.fill_between(cov, mu - sd, mu + sd, color=COL[m], alpha=0.2)
        ax.set_xlabel("coverage (fraction answered, rest abstained)")
        ax.set_ylabel(yl)
axs[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIGURES / "risk_coverage.png", dpi=160)

g = audit_groups(audit)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
names = ["present_correct", "present_wrong"]
x = np.arange(2)
ax.bar(x - 0.18, [g[n]["flip_cited"] for n in names], 0.36, color=COL["agent"], label="delete the cited sentence")
ax.bar(
    x + 0.18,
    [g[n]["flip_random"] for n in names],
    0.36,
    color=COL["keyword_anywhere"],
    label="delete a random other sentence",
)
ax.set_xticks(
    x,
    [
        f"confident 'present',\ncorrect (n={g['present_correct']['n']})",
        f"confident 'present',\nwrong (n={g['present_wrong']['n']})",
    ],
)
ax.set_ylabel("decision flips")
ax.set_ylim(0, 1.1)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIGURES / "evidence_audit.png", dpi=160)
