"""Parse the Open-i / Indiana University chest X-ray report XML files into one JSONL file.

Labels are the *manually assigned* MeSH ``<major>`` tags of the collection (NLM indexers), not tags
produced by an automatic tagger. Only reports with a non-empty FINDINGS or IMPRESSION section are kept.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from .common import RAW, REPORTS, URL

# finding name -> MeSH major tags that count as "present" (lower-case, before the first '/')
LABEL_TAGS = {
    "cardiomegaly": {"cardiomegaly"},
    "pleural effusion": {"pleural effusion"},
    "atelectasis": {"pulmonary atelectasis"},
    "opacity": {"opacity"},
    "pulmonary edema": {"pulmonary edema"},
    "pneumonia": {"pneumonia"},
    "emphysema": {"emphysema", "pulmonary emphysema"},
    "nodule": {"nodule"},
}


def download() -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    tgz = RAW / "NLMCXR_reports.tgz"
    if not tgz.exists():
        urllib.request.urlretrieve(URL, tgz)  # 1.1 MB
    out = RAW / "ecgen-radiology"
    if not out.exists():
        with tarfile.open(tgz) as t:
            for m in t.getmembers():  # refuse path traversal
                if m.name.startswith(("/", "..")) or ".." in Path(m.name).parts:
                    raise ValueError(m.name)
            t.extractall(RAW, filter="data")
    return out


def parse_report(path: Path) -> dict | None:
    root = ET.parse(path).getroot()
    sections = {}
    for el in root.iter("AbstractText"):
        sections[(el.get("Label") or "").lower()] = re.sub(r"\s+", " ", (el.text or "").strip())
    tags = []
    for mesh in root.iter("MeSH"):
        for maj in mesh.findall("major"):
            tags.append((maj.text or "").split("/")[0].strip().lower())
    findings, impression = sections.get("findings", ""), sections.get("impression", "")
    if not findings and not impression:
        return None
    return {
        "id": path.stem,
        "indication": sections.get("indication", ""),
        "comparison": sections.get("comparison", ""),
        "findings": findings,
        "impression": impression,
        "tags": sorted(set(tags)),
    }


def labels(rec: dict) -> dict[str, int]:
    tags = set(rec["tags"])
    return {f: int(bool(tags & LABEL_TAGS[f])) for f in LABEL_TAGS}


def build() -> list[dict]:
    src = download()
    recs = [r for p in sorted(src.glob("*.xml"), key=lambda p: int(p.stem)) if (r := parse_report(p))]
    REPORTS.parent.mkdir(parents=True, exist_ok=True)
    with REPORTS.open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return recs


def load() -> list[dict]:
    return [json.loads(line) for line in REPORTS.read_text().splitlines()]


def split_ids(recs: list[dict], seed: int, val_frac: float, test_frac: float) -> dict[str, list[int]]:
    """Deterministic seeded split by hashing the report id (no patient id exists in the public files)."""
    out = {"train": [], "val": [], "test": []}
    for i, r in enumerate(recs):
        h = int(hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        out["test" if h < test_frac else "val" if h < test_frac + val_frac else "train"].append(i)
    return out
