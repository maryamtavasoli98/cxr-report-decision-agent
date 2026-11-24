# Reproduce everything: make venv data eval figures readme
PY ?= .venv/bin/python

.PHONY: venv data eval figures readme all test lint

venv:
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ".[dev]"

data:            ## download the Open-i report XML (1.1 MB) and write data/reports.jsonl + dataset.json
	$(PY) scripts/prepare_data.py

eval:            ## 4 methods x 8 findings x 5 splits, selective prediction, evidence audit (about 15 seconds on a laptop CPU)
	$(PY) scripts/run_eval.py

figures:
	$(PY) scripts/make_figures.py

readme:          ## fill the README / report tables from results/metrics
	$(PY) scripts/update_readme.py

all: data eval figures readme

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .
