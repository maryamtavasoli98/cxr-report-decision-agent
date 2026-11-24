#!/usr/bin/env python3
"""Run the three experiments and write results/metrics/*.json. Everything is CPU and takes a few minutes."""

import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from cxr_report_decision_agent import tools
from cxr_report_decision_agent.agent import FEATURES, DecisionAgent, run_tools
from cxr_report_decision_agent.common import config, environment, write_json
from cxr_report_decision_agent.data import labels, load, split_ids
from cxr_report_decision_agent.evaluate import aurc, bootstrap_ci, prf, risk_coverage

cfg = config()
F = cfg["findings"]
TAU = cfg["confident_threshold"]
GRID = np.round(np.arange(0.3, 1.0001, 0.05), 2)
recs = load()
N = len(recs)
Y = {f: np.array([labels(r)[f] for r in recs]) for f in F}
text = [(r["findings"] + " " + r["impression"]).strip() for r in recs]

t0 = time.perf_counter()
X, TR = {f: np.zeros((N, len(FEATURES))) for f in F}, {f: [] for f in F}
for f in F:
    for i, r in enumerate(recs):
        X[f][i], tr = run_tools(r, f)
        TR[f].append(tr)
tool_seconds = time.perf_counter() - t0
calls = np.array([[len(t.calls) for t in TR[f]] for f in F])
print(f"tools done in {tool_seconds:.1f}s; mean calls/decision {calls.mean():.2f}", flush=True)

# non-learned baselines
B0 = {f: np.array([int(tools.keyword_in(text[i], f)) for i in range(N)]) for f in F}  # keyword anywhere
B1 = {f: X[f][:, FEATURES.index("any_aff")].astype(int) for f in F}  # tools + fixed rule, no learning

Cs = [0.1, 1.0, 10.0]
THR = np.round(np.arange(0.05, 0.96, 0.05), 2)
per_seed, rc_all, op_all = [], [], []
for seed in cfg["seeds"]:
    sp = split_ids(recs, seed, cfg["val_frac"], cfg["test_frac"])
    tr_i, va_i, te_i = (np.array(sp[k]) for k in ("train", "val", "test"))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([text[i] for i in tr_i])
    T = vec.transform(text)
    out = {"seed": seed, "n_test": len(te_i), "findings": {}}
    pairs = {m: {"p": [], "y": [], "pred": []} for m in ("agent", "tfidf")}
    ap_by_C = []
    for C in Cs:
        a = DecisionAgent(F, C=C).fit({f: X[f][tr_i] for f in F}, {f: Y[f][tr_i] for f in F})
        ap_by_C.append(np.mean([average_precision_score(Y[f][va_i], a.proba(f, X[f][va_i])) for f in F]))
    C_agent = Cs[int(np.argmax(ap_by_C))]
    ag = DecisionAgent(F, C=C_agent).fit({f: X[f][tr_i] for f in F}, {f: Y[f][tr_i] for f in F})
    for f in F:
        y = Y[f]
        res = {}
        # choose C on validation average precision, per model family
        scores = []
        for C in Cs:
            m = LogisticRegression(C=C, max_iter=2000).fit(T[tr_i], y[tr_i])
            scores.append(average_precision_score(y[va_i], m.predict_proba(T[va_i])[:, 1]))
        best = {"agent": C_agent, "tfidf": Cs[int(np.argmax(scores))]}
        p_ag = ag.proba(f, X[f][te_i])
        pv_ag = ag.proba(f, X[f][va_i])
        tf = LogisticRegression(C=best["tfidf"], max_iter=2000).fit(T[tr_i], y[tr_i])
        p_tf = tf.predict_proba(T[te_i])[:, 1]
        pv_tf = tf.predict_proba(T[va_i])[:, 1]
        res["keyword_anywhere"] = prf(y[te_i], B0[f][te_i])
        res["tools_fixed_rule"] = prf(y[te_i], B1[f][te_i])
        for name, p, pv in (("agent", p_ag, pv_ag), ("tfidf", p_tf, pv_tf)):
            thr = float(max(THR, key=lambda t: prf(y[va_i], pv >= t)["f1"]))  # decision cutoff tuned on validation F1
            res[name] = prf(y[te_i], p >= thr)
            res[name]["threshold"] = thr
            res[name]["auroc"] = float(roc_auc_score(y[te_i], p))
            res[name]["ap"] = float(average_precision_score(y[te_i], p))
            res[name]["C"] = best[name]
            pairs[name]["p"].append(p)
            pairs[name]["y"].append(y[te_i])
            pairs[name]["pred"].append((p >= 0.5).astype(int))  # selective prediction uses the 0.5 cutoff
        res["n_pos_test"] = int(y[te_i].sum())
        out["findings"][f] = res
    # experiment 2: pooled risk-coverage
    rc = {}
    for name in pairs:
        p, y, pred = (np.concatenate(pairs[name][k]) for k in ("p", "y", "pred"))
        conf, correct = np.maximum(p, 1 - p), (pred == y).astype(float)
        rc[name] = {
            "curve": risk_coverage(conf, correct, y, pred, GRID),
            "aurc": aurc(conf, correct),
            "full_error": float(1 - correct.mean()),
        }
        ans = conf >= TAU
        m = prf(y[ans], pred[ans])
        rc[name]["operating_point"] = {
            "tau": TAU,
            "coverage": float(ans.mean()),
            "error_rate_answered": float(1 - correct[ans].mean()),
            "micro_f1_answered": m["f1"],
            "positives_abstained": float(1 - y[ans].sum() / y.sum()),
            "n_pairs": int(len(y)),
        }
    out["selective"] = rc
    per_seed.append(out)
    print("seed", seed, "done", flush=True)

write_json(
    "classification.json",
    {
        "config": cfg,
        "findings": F,
        "seeds": per_seed,
        "tool_seconds_all_decisions": tool_seconds,
        "mean_tool_calls_per_decision": float(calls.mean()),
        "n_decisions": int(calls.size),
        "environment": environment(),
    },
)

# bootstrap CI for macro-F1 difference agent - tfidf, seed 0 test set (resampling reports)
sp = split_ids(recs, 0, cfg["val_frac"], cfg["test_frac"])
tr_i, te_i = np.array(sp["train"]), np.array(sp["test"])
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([text[i] for i in tr_i])
T = vec.transform(text)
pred = {"agent": {}, "tfidf": {}}
agent = DecisionAgent(F, C=per_seed[0]["findings"][F[0]]["agent"]["C"]).fit(
    {f: X[f][tr_i] for f in F}, {f: Y[f][tr_i] for f in F}
)
for f in F:
    C_t = per_seed[0]["findings"][f]["tfidf"]["C"]
    pred["agent"][f] = (agent.proba(f, X[f][te_i]) >= per_seed[0]["findings"][f]["agent"]["threshold"]).astype(int)
    pred["tfidf"][f] = (
        LogisticRegression(C=C_t, max_iter=2000).fit(T[tr_i], Y[f][tr_i]).predict_proba(T[te_i])[:, 1]
        >= per_seed[0]["findings"][f]["tfidf"]["threshold"]
    ).astype(int)


def macro(name, idx):
    return float(np.mean([prf(Y[f][te_i][idx], pred[name][f][idx])["f1"] for f in F]))


diff = bootstrap_ci(lambda idx: macro("agent", idx) - macro("tfidf", idx), len(te_i), cfg["bootstrap"])
write_json(
    "bootstrap.json",
    {
        "seed": 0,
        "macro_f1_agent_minus_tfidf": {"point": diff[0], "lo95": diff[1], "hi95": diff[2]},
        "n_test_reports": int(len(te_i)),
        "n_boot": cfg["bootstrap"],
    },
)

# experiment 3: evidence-deletion audit on the seed-0 test set
rng = np.random.default_rng(0)
rows, examples = [], []
for f in F:
    for j, i in enumerate(te_i):
        r, x, tr = recs[i], X[f][i], TR[f][i]
        p = float(agent.proba(f, x)[0])
        act = DecisionAgent.execute(p, TAU)
        if act == "abstain" or tr.evidence is None:
            continue
        ev = tr.evidence
        truth = int(Y[f][i])
        correct = (act == "present") == bool(truth)
        sents = tools.split_sentences(r[ev["section"]])
        others = [k for k in range(len(sents)) if k != ev["sentence_idx"]]
        xd, trd = run_tools(r, f, drop=(ev["section"], ev["sentence_idx"]))
        act_d = DecisionAgent.execute(float(agent.proba(f, xd)[0]), TAU)
        row = {
            "finding": f,
            "report": r["id"],
            "decision": act,
            "truth": truth,
            "correct": correct,
            "p": p,
            "evidence_context": ev["context"],
            "flip_cited": act_d != act,
        }
        if others:
            xr, _ = run_tools(r, f, drop=(ev["section"], int(rng.choice(others))))
            row["flip_random"] = DecisionAgent.execute(float(agent.proba(f, xr)[0]), TAU) != act
        rows.append(row)
        if not correct:
            examples.append(
                {
                    "finding": f,
                    "report": r["id"],
                    "decision": act,
                    "truth": truth,
                    "p": round(p, 3),
                    "evidence": ev["sentence"],
                    "context": ev["context"],
                    "section": ev["section"],
                    "flip_cited": row["flip_cited"],
                    "tags": r["tags"],
                }
            )
write_json(
    "audit.json",
    {
        "tau": TAU,
        "n_answered": len(rows),
        "rows": rows,
        "confident_errors": sorted(examples, key=lambda e: -abs(e["p"] - 0.5)),
    },
)
# coefficients and timing
t0 = time.perf_counter()
for i in te_i[:200]:
    run_tools(recs[i], "pleural effusion")
write_json(
    "agent_internals.json",
    {
        "coefficients": agent.explain(),
        "features": FEATURES,
        "ms_per_decision_tools": 1000 * (time.perf_counter() - t0) / 200,
    },
)
print("audit rows", len(rows))
