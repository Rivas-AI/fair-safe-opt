.PHONY: test smoke check-env

test:
	python -m pytest

smoke:
	bash scripts/run_smoke.sh

check-env:
	python scripts/check_environment.py

inspect-adult:
	python scripts/inspect_adult.py

run:
	python scripts/run.py

play: 
	python scripts/baseline_v_attack.py