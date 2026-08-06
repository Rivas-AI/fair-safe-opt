.PHONY: test smoke check-env inspect-adult dataset-summary \
	train-baselines train-static-adversarial train-madry-adversarial train-fair-adversarial \
	baseline-robustness adv-robustness eval-fair-adv eval-fair-adv-tau-sweep

test:
	python -m pytest

smoke:
	bash scripts/week01/run_smoke.sh

check-env:
	python scripts/week01/check_environment.py

inspect-adult:
	python scripts/week01/inspect_adult.py

dataset-summary:
	python scripts/dataset_summary.py

train-baselines:
	python scripts/training/train_baselines.py

train-static-adversarial:
	python scripts/training/train_static_adversarial.py

train-madry-adversarial:
	python scripts/training/train_madry_adversarial.py

train-fair-adversarial:
	python scripts/training/train_fair_robust.py

baseline-robustness:
	python scripts/robustness_eval/baseline_robustness_eval.py

adv-robustness:
	python scripts/robustness_eval/adv_robustness_eval.py

eval-fair-adv:
	python scripts/robustness_eval/fair_adv_robustness_eval.py

eval-fair-adv-tau-sweep:
	python scripts/robustness_eval/fair_adv_tolerance_sweep_eval.py
