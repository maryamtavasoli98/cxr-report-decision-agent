# An auditable tool-using decision agent for findings in chest X-ray reports

## Abstract
We build a small decision agent that answers, for eight findings, whether a chest X-ray report states the finding as
present. The agent acts only through tools (impression lookup, findings lookup when needed, a NegEx-style context
assessment), converts the tool output into a probability with one shared logistic head, and answers or abstains by a
fixed rule; each answer cites one sentence. On 3,927 public Indiana University reports with manual MeSH labels, the agent
reaches a macro F1 comparable to a TF-IDF logistic regression, and an evidence-deletion test shows that the cited sentence
is necessary for the answer. A fixed confidence threshold abstains mostly on true positives of rare findings.

## 1. Introduction
Agents that act on clinical text are hard to audit when their reasoning is free text. Bao et al. [1] (Truhn among the
authors) describe Decision2Care, which assigns a probability to every permitted action and lets fixed rules execute the most
probable one; according to the abstract, confident errors were traced to single imaging reports. We ask a smaller question
on public data and a deliberately simple agent: can a tool-and-rule agent match a plain classifier on report understanding,
and can its citations be tested rather than trusted? This is not a reproduction of [1]; only its abstract was read.

## 2. Related work
Report labelling with rules and an uncertainty class is established practice [4]; negation scope in clinical text was
handled by NegEx [3], whose idea (trigger terms with a limited scope) we re-implement. Abstention is studied as selective
classification with risk-coverage curves [5]. The report collection is described in [2].

## 3. Method
**Data.** Reports from the Open-i Indiana University collection [2], 3,927 with a non-empty FINDINGS or IMPRESSION section.
Label = the finding's manually assigned MeSH major tag is present. Splits are 60/20/20 by hashed report id, five seeds.

<!-- table:data -->
| finding | reports with MeSH tag | share |
|---|---|---|
| cardiomegaly | 375 | 9.5 % |
| pleural effusion | 161 | 4.1 % |
| atelectasis | 332 | 8.5 % |
| opacity | 455 | 11.6 % |
| pulmonary edema | 46 | 1.2 % |
| pneumonia | 42 | 1.1 % |
| emphysema | 98 | 2.5 % |
| nodule | 111 | 2.8 % |
| **all reports used** | 3927 | |
<!-- /table:data -->

**Tools.** `find_mentions(section, finding)` matches a hand-written lexicon per finding and returns sentences with a context
flag from `assess_context` (negated, uncertain, historical). **Policy.** Query the impression; query the findings only if the
impression has no affirmed mention; query the indication only if there is no mention at all. **Head.** One logistic
regression over 13 count features plus per-finding intercepts and slopes on two summary features, with p(present).
**Executor.** Answer if max(p, 1-p) is at least 0.9, else abstain. **Baselines.** B0 keyword anywhere; B1 same tools with
the rule "any affirmed mention"; B3 TF-IDF logistic regression per finding. C and the cutoff are chosen on validation data.
**Audit.** For each confident answer, delete the cited sentence (or a random other sentence of the same section), rerun
the agent, and record whether the answer changes.

## 4. Experiments
**Classification.** Macro F1 over eight findings, mean ± sd over five splits.

<!-- table:main -->
| method | macro F1 | macro precision | macro recall |
|---|---|---|---|
| B0 keyword anywhere (no negation) | 0.424 ± 0.016 | 0.320 ± 0.012 | 0.985 ± 0.001 |
| B1 tools + fixed rule (no learning) | 0.696 ± 0.018 | 0.613 ± 0.019 | 0.896 ± 0.025 |
| B2 decision agent (tools + learned head) | 0.719 ± 0.029 | 0.648 ± 0.027 | 0.856 ± 0.035 |
| B3 TF-IDF logistic regression (no tools) | 0.716 ± 0.052 | 0.775 ± 0.045 | 0.713 ± 0.053 |

Agent minus TF-IDF, macro F1, split seed 0 (796 test reports, 1000 bootstrap resamples): -0.020 (95 % CI -0.094 to +0.053).
<!-- /table:main -->

<!-- table:per_finding -->
| finding | B0 keyword | B1 tools + rule | B2 agent | B3 TF-IDF | B2 AUROC | B3 AUROC |
|---|---|---|---|---|---|---|
| cardiomegaly | 0.82 | 0.83 | 0.84 | 0.92 | 0.960 | 0.997 |
| pleural effusion | 0.12 | 0.85 | 0.84 | 0.72 | 0.981 | 0.977 |
| atelectasis | 0.89 | 0.92 | 0.91 | 0.87 | 0.991 | 0.992 |
| opacity | 0.30 | 0.76 | 0.79 | 0.84 | 0.965 | 0.989 |
| pulmonary edema | 0.23 | 0.70 | 0.71 | 0.47 | 0.995 | 0.988 |
| pneumonia | 0.27 | 0.68 | 0.68 | 0.67 | 0.996 | 0.996 |
| emphysema | 0.49 | 0.52 | 0.54 | 0.67 | 0.966 | 0.979 |
| nodule | 0.28 | 0.31 | 0.44 | 0.57 | 0.954 | 0.959 |
<!-- /table:per_finding -->

**Selective prediction.**

<!-- table:selective -->
| method | coverage at confidence ≥ 0.9 | error rate on answered pairs | positive-class micro F1 on answered | true positives abstained on |
|---|---|---|---|---|
| B2 decision agent (tools + learned head) | 0.932 ± 0.009 | 0.0060 ± 0.0007 | 0.774 ± 0.074 | 0.720 ± 0.083 |
| B3 TF-IDF logistic regression (no tools) | 0.933 ± 0.002 | 0.0060 ± 0.0011 | 0.766 ± 0.040 | 0.733 ± 0.029 |
<!-- /table:selective -->

**Evidence deletion and errors.**

<!-- table:audit -->
| confident decision (agent, seed-0 test set) | n | flips when the cited sentence is deleted | flips when a random other sentence of the same section is deleted |
|---|---|---|---|
| answered 'present', correct | 55 | 93 % | 4 % (n=47) |
| answered 'present', wrong | 4 | 75 % | 25 % (n=4) |

Confident 'absent' answers: 1106 correct, 22 wrong (3 cited a sentence judged 'negated', 19 cited a sentence judged 'uncertain').
<!-- /table:audit -->

<!-- table:errors -->
| finding | answered | truth | p(present) | cited sentence (section, context) | flips on deletion |
|---|---|---|---|---|---|
| pleural effusion | absent | present | 0.003 | No focal alveolar consolidation, no definite pleural effusion seen. (findings, negated) | no |
| pleural effusion | absent | present | 0.004 | No significant change in right pneumothorax or pleural fluid. (impression, negated) | no |
| opacity | absent | present | 0.008 | No XXXX pulmonary consolidation. (impression, negated) | no |
| pleural effusion | present | absent | 0.972 | Blunting of left costophrenic XXXX. (impression, affirmed) | yes |
| pulmonary edema | absent | present | 0.028 | Marked bronchovascular crowding, indistinct vascular margination may be secondary to crowding, mild pulmonary edema, interstitial infiltrates difficult to exclude. (impression, uncertain) | no |
| nodule | absent | present | 0.033 | Dense nodule again noted in the right lower lobe suggesting a previous granulomatous process. (findings, uncertain) | no |
<!-- /table:errors -->

**Cost.** <!-- table:env -->
Python 3.13.7, numpy 2.5.5, scikit-learn 1.9.1, arm64 / Darwin, CPU only. All 31,416 tool-pipeline runs (8 findings x 3,927 reports) took 0.5 s; 0.02 ms per decision; 2.70 tool calls per decision on average.
<!-- /table:env -->

The learned head is a small improvement over the fixed rule on macro F1, not tested for significance. The agent and TF-IDF
are statistically indistinguishable on macro F1 in the paired bootstrap; each is better on different findings. The audit shows the cited
sentence is necessary for confident "present" answers. Most confident "absent" errors cite a hedged mention that the
indexers nevertheless tagged as present, and a few labels contradict the report text.

## 5. Limitations
Labels are MeSH indexing, not image-based findings. Lexicons and negation rules are hand-made; the head design was revised
after seeing seed-0 test results, so agent numbers are slightly optimistic. Rare findings have few positives. The agent is
rule-driven, not an LLM, so nothing here supports claims about LLM agents. The audit has few confident wrong answers.
Report-level splits may leak templated reports. A fixed 0.9 threshold needs per-finding calibration.

## 6. Conclusion
A tool-and-rule agent with a learned probability head matches a text classifier on this task and supports a testable
citation audit. The next steps would be a calibrated per-finding abstention rule, an LLM-driven policy under the same
executor, and image-grounded labels.

## References

[1] T. Bao et al., Auditable Clinical Support with Decision Agents, arXiv:2611.23700, 2026.
[2] D. Demner-Fushman et al., Preparing a collection of radiology examinations for distribution and retrieval, JAMIA, 2015, doi:10.1093/jamia/ocv080.
[3] W. W. Chapman et al., A Simple Algorithm for Identifying Negated Findings and Diseases in Discharge Summaries, J. Biomed. Inform., 2001, doi:10.1006/jbin.2001.1029.
[4] J. Irvin et al., CheXpert, AAAI 2019, doi:10.1609/aaai.v33i01.3301590.
[5] Y. Geifman, R. El-Yaniv, Selective Classification for Deep Neural Networks, arXiv:1705.08500, 2017.
