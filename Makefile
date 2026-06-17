.PHONY: test smoke check-env

test:
	python -m pytest

smoke:
	bash scripts/run_smoke.sh

check-env:
	python scripts/check_environment.py

inspect-adult:
	python scripts/inspect_adult.py

report:
	python scripts/train_and_generate_report.py