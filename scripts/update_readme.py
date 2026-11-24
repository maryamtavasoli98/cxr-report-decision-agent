#!/usr/bin/env python3
"""Fill the ``<!-- table:NAME -->`` blocks of README.md and report/report.md from results/metrics (nothing typed by hand)."""

import re

from cxr_report_decision_agent.common import ROOT, read_json
from cxr_report_decision_agent.summary import LABEL, METHODS, audit_groups, macro_f1, macro_key, operating, per_finding

TARGETS = [ROOT / "README.md", ROOT / "report" / "report.md"]
cls, audit = read_json("classification.json"), read_json("audit.json")
boot, ds, internals = read_json("bootstrap.json"), read_json("dataset.json"), read_json("agent_internals.json")
F = cls["findings"]


def pm(t, d=3):
    return f"{t[0]:.{d}f} ± {t[1]:.{d}f}"


def table_data():
    rows = ["| finding | reports with MeSH tag | share |", "|---|---|---|"]
    for f in F:
        n = ds["positives_per_finding"][f]
        rows.append(f"| {f} | {n} | {100 * n / ds['n_reports']:.1f} % |")
    rows.append(f"| **all reports used** | {ds['n_reports']} | |")
    return "\n".join(rows)


def table_main():
    rows = ["| method | macro F1 | macro precision | macro recall |", "|---|---|---|---|"]
    for m in METHODS:
        rows.append(
            f"| {LABEL[m]} | {pm(macro_f1(cls, m))} | {pm(macro_key(cls, m, 'precision'))} | {pm(macro_key(cls, m, 'recall'))} |"
        )
    d = boot["macro_f1_agent_minus_tfidf"]
    rows.append("")
    rows.append(
        f"Agent minus TF-IDF, macro F1, split seed 0 ({boot['n_test_reports']} test reports, {boot['n_boot']} bootstrap resamples): "
        f"{d['point']:+.3f} (95 % CI {d['lo95']:+.3f} to {d['hi95']:+.3f})."
    )
    return "\n".join(rows)


def table_per_finding():
    rows = [
        "| finding | B0 keyword | B1 tools + rule | B2 agent | B3 TF-IDF | B2 AUROC | B3 AUROC |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in F:
        rows.append(
            f"| {f} | "
            + " | ".join(f"{per_finding(cls, f, m)[0]:.2f}" for m in METHODS)
            + f" | {per_finding(cls, f, 'agent', 'auroc')[0]:.3f} | {per_finding(cls, f, 'tfidf', 'auroc')[0]:.3f} |"
        )
    return "\n".join(rows)


def table_selective():
    tau = cls["config"]["confident_threshold"]
    rows = [
        f"| method | coverage at confidence ≥ {tau} | error rate on answered pairs | positive-class micro F1 on answered | true positives abstained on |",
        "|---|---|---|---|---|",
    ]
    for m in ("agent", "tfidf"):
        rows.append(
            f"| {LABEL[m]} | {pm(operating(cls, m, 'coverage'))} | {pm(operating(cls, m, 'error_rate_answered'), 4)} | "
            f"{pm(operating(cls, m, 'micro_f1_answered'))} | {pm(operating(cls, m, 'positives_abstained'))} |"
        )
    return "\n".join(rows)


def table_audit():
    g = audit_groups(audit)
    rows = [
        "| confident decision (agent, seed-0 test set) | n | flips when the cited sentence is deleted | flips when a random other sentence of the same section is deleted |",
        "|---|---|---|---|",
    ]
    for key, name in (
        ("present_correct", "answered 'present', correct"),
        ("present_wrong", "answered 'present', wrong"),
    ):
        r = g[key]
        rows.append(
            f"| {name} | {r['n']} | {100 * r['flip_cited']:.0f} % | {100 * r['flip_random']:.0f} % (n={r['n_control']}) |"
        )
    aw = g["absent_wrong"]
    ctx = ", ".join(f"{v} cited a sentence judged '{k}'" for k, v in sorted(aw["by_context"].items()))
    rows.append("")
    rows.append(f"Confident 'absent' answers: {g['absent_correct_n']} correct, {aw['n']} wrong ({ctx}).")
    return "\n".join(rows)


def table_errors():
    rows = [
        "| finding | answered | truth | p(present) | cited sentence (section, context) | flips on deletion |",
        "|---|---|---|---|---|---|",
    ]
    for e in audit["confident_errors"][:6]:
        s = e["evidence"].replace("|", "/")
        rows.append(
            f"| {e['finding']} | {e['decision']} | {'present' if e['truth'] else 'absent'} | {e['p']:.3f} | {s} ({e['section']}, {e['context']}) | {'yes' if e['flip_cited'] else 'no'} |"
        )
    return "\n".join(rows)


def table_env():
    e = cls["environment"]
    return (
        f"Python {e['python']}, numpy {e['numpy']}, scikit-learn {e['scikit_learn']}, {e['machine']} / {e['platform']}, {e['compute']}. "
        f"All {cls['n_decisions']:,} tool-pipeline runs (8 findings x {ds['n_reports']:,} reports) took {cls['tool_seconds_all_decisions']:.1f} s; "
        f"{internals['ms_per_decision_tools']:.2f} ms per decision; {cls['mean_tool_calls_per_decision']:.2f} tool calls per decision on average."
    )


def table_coef():
    c = internals["coefficients"]
    shared = [(k, v) for k, v in c.items() if k in internals["features"]]
    rows = ["| tool feature | weight (seed-0 head) |", "|---|---|"]
    for k, v in sorted(shared, key=lambda kv: -abs(kv[1])):
        rows.append(f"| {k} | {v:+.2f} |")
    return "\n".join(rows)


for path in TARGETS:
    if not path.exists():
        continue
    txt = path.read_text()
    for name in re.findall(r"<!-- table:(\w+) -->", txt):
        body = globals()[f"table_{name}"]()
        txt = re.sub(
            rf"(<!-- table:{name} -->).*?(<!-- /table:{name} -->)",
            lambda m: f"{m.group(1)}\n{body}\n{m.group(2)}",
            txt,
            flags=re.S,
        )
    path.write_text(txt)
print("readme updated")
