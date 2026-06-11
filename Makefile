.PHONY: test smoke check-env

test:
	python -m pytest

smoke:
	bash scripts/run_smoke.sh

check-env:
	python scripts/check_environment.py

inspect-adult:
	python scripts/inspect_adult.py

baseline-logistic:
	python scripts/run_baseline_logistic.py

baseline-mlp:
	python scripts/baseline_mlp.py