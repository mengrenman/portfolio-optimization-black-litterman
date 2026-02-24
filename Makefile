.PHONY: setup test run-buffett run-pelosi run-trump lint notebook

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
