import numpy as np

from cxr_report_decision_agent import tools
from cxr_report_decision_agent.agent import FEATURES, DecisionAgent, run_tools
from cxr_report_decision_agent.data import split_ids
from cxr_report_decision_agent.evaluate import aurc, prf, risk_coverage


def ctx(sentence, term):
    i = sentence.lower().index(term)
    return tools.assess_context(sentence, (i, i + len(term)))


def test_negation_list_scope():
    s = "No focal consolidation, pleural effusion, or pneumothorax."
    assert ctx(s, "pleural effusion")[0] is True


def test_affirmed():
    assert ctx("There is a small left pleural effusion.", "pleural effusion") == (False, False, False)


def test_terminator_stops_scope():
    s = "No pneumothorax, but there is a pleural effusion."
    assert ctx(s, "pleural effusion")[0] is False


def test_uncertain_and_historical():
    n, u, h = ctx("Possible small effusion, unchanged.", "effusion")
    assert (n, u, h) == (False, True, True)


def test_not_excluded_is_not_negation():
    assert ctx("Pneumonia cannot be excluded.", "pneumonia") == (False, True, False)


def test_xxxx_token_negation():
    assert ctx("There are no XXXX of a pleural effusion.", "pleural effusion")[0] is True


def test_sentence_split():
    assert len(tools.split_sentences("Heart is normal. Lungs are clear. XXXX seen.")) == 3


def test_run_tools_calls_second_tool_only_when_needed():
    rep = {
        "impression": "Small left pleural effusion.",
        "findings": "Heart normal.",
        "indication": "",
        "comparison": "",
    }
    x, tr = run_tools(rep, "pleural effusion")
    assert x[FEATURES.index("imp_aff")] == 1 and x[FEATURES.index("find_called")] == 0 and len(tr.calls) == 1
    rep2 = {"impression": "Normal.", "findings": "Mild cardiomegaly.", "indication": "", "comparison": ""}
    x2, tr2 = run_tools(rep2, "cardiomegaly")
    assert x2[FEATURES.index("find_called")] == 1 and tr2.evidence["section"] == "findings"


def test_deleting_cited_sentence_removes_decision():
    rep = {"impression": "Small left pleural effusion.", "findings": "", "indication": "", "comparison": ""}
    x, tr = run_tools(rep, "pleural effusion")
    xd, _ = run_tools(rep, "pleural effusion", drop=("impression", tr.evidence["sentence_idx"]))
    assert x[FEATURES.index("any_aff")] == 1 and xd[FEATURES.index("any_aff")] == 0


def test_drop_keeps_original_sentence_index():
    rep = {"impression": "Heart normal. Small effusion.", "findings": "", "indication": "", "comparison": ""}
    _, tr = run_tools(rep, "pleural effusion", drop=("impression", 0))
    assert tr.evidence["sentence_idx"] == 1


def test_execute_rule():
    assert DecisionAgent.execute(0.97, 0.9) == "present"
    assert DecisionAgent.execute(0.02, 0.9) == "absent"
    assert DecisionAgent.execute(0.6, 0.9) == "abstain"


def test_prf():
    m = prf(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1) and abs(m["f1"] - 0.5) < 1e-9


def test_risk_coverage_and_aurc_perfect_ranking():
    conf = np.array([0.99, 0.9, 0.6, 0.55])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    y = np.array([1, 0, 1, 0])
    rc = risk_coverage(conf, correct, y, y, np.array([0.5, 1.0]))
    assert rc[0]["error_rate"] == 0.0 and abs(rc[1]["error_rate"] - 0.5) < 1e-9
    assert aurc(conf, correct) < aurc(conf[::-1], correct)


def test_split_is_deterministic_and_disjoint():
    recs = [{"id": str(i)} for i in range(500)]
    a, b = split_ids(recs, 0, 0.2, 0.2), split_ids(recs, 0, 0.2, 0.2)
    assert a == b
    assert len(set(a["train"]) | set(a["val"]) | set(a["test"])) == 500
    assert not set(a["test"]) & set(a["train"])
