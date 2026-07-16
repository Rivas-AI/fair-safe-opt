.PHONY: test smoke check-env

test:
	python -m pytest

smoke:
	bash scripts/run_smoke.sh

check-env:
	python scripts/check_environment.py

inspect-adult:
	python scripts/inspect_adult.py

train_baselines:
	python scripts/train_baselines.py

baseline_robustness:
	python scripts/baseline_robustness_eval.py

train_static_adversarial:
	python scripts/train_static_adversarial.py

adv_robustness:
	python scripts/adv_robustness_eval.py

train_madry_adversarial:
	python scripts/train_madry_adversarial.py