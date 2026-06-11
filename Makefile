.PHONY: test smoke check-env

test:
	python -m pytest

smoke:
	bash scripts/run_smoke.sh

check-env:
	python scripts/check_environment.py

split-inspect:
	python scripts/inspect_split.py

logistic:
	python scripts/run_logistic.py