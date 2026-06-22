.PHONY: test smoke check-env

test:
	python -m pytest

smoke:
	bash scripts/run_smoke.sh

check-env:
	python scripts/check_environment.py

inspect-adult:
	python scripts/inspect_adult.py

baseline:
	python scripts/generate_baseline_report.py

attack:
	python scripts/generate_attack_report.py