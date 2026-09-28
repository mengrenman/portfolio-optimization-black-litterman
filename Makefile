.PHONY: setup test run-buffett run-pelosi run-trump lint notebook attribution tables

setup:
	python -m pip install -e '.[dev,notebooks]'

test:
	pytest -q

lint:
	ruff check src tests scripts

run-buffett:
	python scripts/run_case_study.py --person buffett

run-pelosi:
	python scripts/run_case_study.py --person pelosi

run-trump:
	python scripts/run_case_study.py --person trump

notebook:
	jupyter notebook

attribution:
	python scripts/factor_attribution.py

tables:
	python scripts/performance_tables.py
