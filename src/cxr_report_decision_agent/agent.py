"""The decision agent. It never writes free text. Control flow is a fixed rule set (call tools in order, stop when
resolved); the only learned part is a small per-finding logistic head that turns tool outputs into a probability,
and a threshold rule that answers or abstains. Every step is logged in a trace so a reviewer can audit it.

This is a small-scale analogue of the structure described for Decision2Care (Bao et al. 2026): probabilities
over permitted actions, executed by fixed rules, with a traceable reason for each decision. It is not a
reimplementation of that system."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import tools

FEATURES = [
    "imp_aff",
    "imp_neg",
    "imp_unc",
    "imp_hist_aff",
    "find_called",
    "find_aff",
    "find_neg",
    "find_unc",
    "find_hist_aff",
    "ind_mention",
    "any_aff",
    "any_neg_only",
    "imp_empty",
]


@dataclass
class Trace:
    calls: list[dict] = field(default_factory=list)
    evidence: dict | None = None
    seconds: float = 0.0


def run_tools(report: dict, finding: str, drop: tuple[str, int] | None = None) -> tuple[np.ndarray, Trace]:
    """Run the tool plan on one report. ``drop=(section, sentence_idx)`` deletes one sentence first (audit)."""
    t0 = time.perf_counter()
    tr = Trace()
    mentions: list[tools.Mention] = []

    def lookup(section: str) -> list[tools.Mention]:
        text = tools.get_section(report, section)
        if drop and drop[0] == section:
            sents = tools.split_sentences(text)
            text = " ".join(s for i, s in enumerate(sents) if i != drop[1])
        # sentence indices must refer to the original report, so re-index when a sentence was dropped
        ms = tools.find_mentions(text, finding, section)
        if drop and drop[0] == section:
            ms = [
                tools.Mention(
                    m.section,
                    m.sentence_idx + (m.sentence_idx >= drop[1]),
                    m.sentence,
                    m.span,
                    m.negated,
                    m.uncertain,
                    m.historical,
                )
                for m in ms
            ]
        tr.calls.append({"tool": "find_mentions", "section": section, "n": len(ms)})
        return ms

    imp = lookup("impression")
    mentions += imp
    f = {k: 0.0 for k in FEATURES}
    f["imp_empty"] = float(not report["impression"])
    for m in imp:
        f["imp_aff"] += m.affirmed
        f["imp_neg"] += m.negated
        f["imp_unc"] += m.uncertain
        f["imp_hist_aff"] += m.affirmed and m.historical
    fnd: list[tools.Mention] = []
    if not f["imp_aff"]:  # the impression did not settle it: call the second tool
        fnd = lookup("findings")
        f["find_called"] = 1.0
        mentions += fnd
        for m in fnd:
            f["find_aff"] += m.affirmed
            f["find_neg"] += m.negated
            f["find_unc"] += m.uncertain
            f["find_hist_aff"] += m.affirmed and m.historical
    if not mentions:  # last resort: does the indication text mention it?
        f["ind_mention"] = float(tools.keyword_in(report["indication"], finding))
        tr.calls.append({"tool": "keyword_in", "section": "indication", "hit": bool(f["ind_mention"])})
    f["any_aff"] = float(any(m.affirmed for m in mentions))
    f["any_neg_only"] = float(bool(mentions) and not f["any_aff"] and all(m.negated for m in mentions))
    # evidence = first affirmed mention, else first uncertain, else first negated mention
    for pick in (lambda m: m.affirmed, lambda m: m.uncertain, lambda m: m.negated):
        hit = next((m for m in mentions if pick(m)), None)
        if hit:
            tr.evidence = {
                "section": hit.section,
                "sentence_idx": hit.sentence_idx,
                "sentence": hit.sentence,
                "context": "affirmed" if hit.affirmed else "uncertain" if hit.uncertain else "negated",
            }
            break
    tr.seconds = time.perf_counter() - t0
    return np.array([f[k] for k in FEATURES]), tr


class DecisionAgent:
    """One probability head shared by all findings: logistic regression over the tool features plus per-finding
    intercepts and per-finding slopes on two summary features. The tool features mean the same thing for every finding, so rare findings borrow
    statistical strength from common ones. A threshold rule answers or abstains."""

    def __init__(self, findings: list[str], C: float = 1.0):
        self.findings = findings
        self.model = LogisticRegression(C=C, max_iter=2000)

    def _design(self, finding: str, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        onehot = np.zeros((len(x), len(self.findings)))
        onehot[:, self.findings.index(finding)] = 1.0
        # per-finding intercept and per-finding slope on the two summary features (any affirmed / impression affirmed)
        inter = [onehot * x[:, [FEATURES.index(k)]] for k in ("any_aff", "imp_aff")]
        return np.hstack([x, onehot, *inter])

    def fit(self, X: dict[str, np.ndarray], y: dict[str, np.ndarray]) -> DecisionAgent:
        A = np.vstack([self._design(f, X[f]) for f in self.findings])
        self.model.fit(A, np.concatenate([y[f] for f in self.findings]))
        return self

    def proba(self, finding: str, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._design(finding, x))[:, 1]

    @staticmethod
    def execute(p: float, tau: float) -> str:
        """Fixed rule executor: answer when confident, otherwise hand over to a human."""
        conf = max(p, 1 - p)
        return "abstain" if conf < tau else "present" if p >= 0.5 else "absent"

    def explain(self) -> dict[str, float]:
        names = (
            FEATURES
            + [f"intercept[{f}]" for f in self.findings]
            + [f"any_aff x {f}" for f in self.findings]
            + [f"imp_aff x {f}" for f in self.findings]
        )
        return dict(zip(names, self.model.coef_[0].round(3).tolist(), strict=True))
