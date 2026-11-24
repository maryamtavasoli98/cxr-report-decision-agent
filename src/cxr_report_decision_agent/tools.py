"""The agent's tools, all written from scratch (no NLP library): sentence splitting, section access,
finding-mention retrieval and a NegEx-style context assessment (negated / uncertain / historical)."""

from __future__ import annotations

import re
from dataclasses import dataclass

# mention patterns per finding (regex, case-insensitive). 'XXXX' is the anonymisation token of the corpus.
MENTION = {
    "cardiomegaly": r"cardiomegaly|cardiac enlargement|enlarged (?:cardiac|heart)|(?:heart|cardiac|cardiomediastinal)[^.;,]{0,30}\b(?:enlarged|enlargement|increased|borderline|upper limits)",
    "pleural effusion": r"pleural effusions?|effusions?|pleural fluid|blunting of (?:the )?(?:left |right )?costophrenic",
    "atelectasis": r"atelecta\w*|subsegmental collapse|(?:lobar|lung|segmental) collapse|discoid",
    "opacity": r"opacit\w+|opacification|consolidat\w*|airspace disease|air ?space (?:disease|opacit\w+)|infiltrat\w*|density|densities",
    "pulmonary edema": r"(?:pulmonary |interstitial )?edema|pulmonary (?:vascular )?congestion|vascular congestion|cephalization",
    "pneumonia": r"pneumonia|pneumonitis|bronchopneumonia",
    "emphysema": r"emphysema\w*|hyperinflat\w*|hyperexpan\w*|copd|chronic obstructive",
    "nodule": r"nodul\w+|nodular|mass(?:es)?\b|granuloma\w*",
}
NEG_PRE = re.compile(
    r"\b(?:no|not|without|negative for|free of|absence of|absent|clear of|ruled out|rule out|resolution of|resolved|"
    r"nor|neither|no evidence of|no definite|no focal|no acute|no XXXX of|denies|devoid of)\b",
    re.I,
)
NEG_POST = re.compile(
    r"\b(?:is|are|was|were|has|have) (?:resolved|absent|ruled out|not seen|not identified|cleared)\b|\bunremarkable\b",
    re.I,
)
NEG_PRE_NOT_NEG = re.compile(
    r"\b(?:not (?:excluded|ruled out)|cannot be (?:excluded|ruled out)|no change|no interval change)\b", re.I
)
UNCERTAIN = re.compile(
    r"\b(?:possible|possibly|probable|probably|may (?:represent|reflect|be)|might|suspicious|suspect(?:ed)?|"
    r"questionable|cannot (?:be )?(?:excluded|ruled out)|not (?:excluded|ruled out)|could (?:represent|be)|versus|vs\.?|"
    r"suggest\w*|concern for|equivocal|raise[sd]? (?:the )?(?:possibility|question)|likely)\b",
    re.I,
)
HISTORICAL = re.compile(
    r"\b(?:stable|unchanged|previous\w*|prior|old|chronic|history of|s/p|status post|healed|remote)\b", re.I
)
TERMINATOR = re.compile(r"\b(?:but|however|although|except|though|aside from|other than)\b|[;:]", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def get_section(report: dict, name: str) -> str:
    return report.get(name, "") or ""


@dataclass
class Mention:
    section: str
    sentence_idx: int
    sentence: str
    span: tuple[int, int]
    negated: bool
    uncertain: bool
    historical: bool

    @property
    def affirmed(self) -> bool:
        return not self.negated and not self.uncertain


def assess_context(sentence: str, span: tuple[int, int]) -> tuple[bool, bool, bool]:
    """NegEx-style: a negation trigger shortly before the term (within 12 tokens, no terminator in between)
    or a post-trigger shortly after negates it; uncertainty cues anywhere in the clause make it uncertain."""
    before, after = sentence[: span[0]], sentence[span[1] :]
    term_end = TERMINATOR.search(after)
    clause_after = after[: term_end.start()] if term_end else after
    # the clause starts after the last terminator before the term
    last = None
    for last in TERMINATOR.finditer(before):
        pass
    clause_before = before[last.end() :] if last else before
    window = " ".join(clause_before.split()[-12:])
    neg_hits = [m for m in NEG_PRE.finditer(window) if not NEG_PRE_NOT_NEG.search(window)]
    negated = bool(neg_hits) or bool(NEG_POST.search(" ".join(clause_after.split()[:8])))
    clause = clause_before + " " + clause_after
    uncertain = bool(UNCERTAIN.search(clause)) and not negated
    historical = bool(HISTORICAL.search(clause))
    return negated, uncertain, historical


def find_mentions(text: str, finding: str, section: str) -> list[Mention]:
    pat = re.compile(MENTION[finding], re.I)
    out = []
    for i, sent in enumerate(split_sentences(text)):
        for m in pat.finditer(sent):
            neg, unc, his = assess_context(sent, m.span())
            out.append(Mention(section, i, sent, m.span(), neg, unc, his))
            break  # one mention per sentence is enough for the evidence trace
    return out


def keyword_in(text: str, finding: str) -> bool:
    return re.search(MENTION[finding], text, re.I) is not None
