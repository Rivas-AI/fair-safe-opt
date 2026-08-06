"""
paths.py
----------------------------------
Single source of truth for on-disk locations.

Every artifact written by a script in scripts/ and read back by evaluation code is
built here, so a writer and its reader cannot drift apart. Paths are anchored to the
repository root rather than the working directory, so importing this module has no
side effects and does not depend on where the interpreter was started.
"""

import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_ROOT = PROJECT_ROOT / "trained_models"
RESULTS_ROOT = PROJECT_ROOT / "results"
DATA_ROOT = PROJECT_ROOT / "data"
TRAINING_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "training_configs.yaml"


def config_path():
    return TRAINING_CONFIG_PATH


def run_dir_name(architecture: str, training_epsilon: float, seed: int) -> str:
    """Leaf directory name for one trained model. Shared by every training style."""
    return f"{architecture}_eps{training_epsilon}_{seed}"


# --- checkpoint trees: written by scripts/train_*.py, read by the evaluation sweeps ---


def baseline_root() -> Path:
    return MODELS_ROOT / "baseline"


def static_adversarial_root() -> Path:
    return MODELS_ROOT / "adversarial" / "static_pgd"


def madry_adversarial_root() -> Path:
    return MODELS_ROOT / "adversarial" / "madry_pgd"


def fair_adversarial_root(metric: str, tolerance: float) -> Path:
    """
    Tolerance is required: a fair-adversarial checkpoint tree cannot be addressed
    without saying which tau it holds.
    """
    return MODELS_ROOT / "fair_adversarial" / metric / f"tau_{tolerance}"


def baseline_run_dir(architecture: str, seed: int) -> Path:
    return baseline_root() / run_dir_name(architecture, 0.0, seed)


def static_adversarial_run_dir(architecture: str, training_epsilon: float, seed: int) -> Path:
    return static_adversarial_root() / run_dir_name(architecture, training_epsilon, seed)


def madry_adversarial_run_dir(architecture: str, training_epsilon: float, seed: int) -> Path:
    return madry_adversarial_root() / run_dir_name(architecture, training_epsilon, seed)


def fair_adversarial_run_dir(
    metric: str, tolerance: float, architecture: str, training_epsilon: float, seed: int
) -> Path:
    return fair_adversarial_root(metric, tolerance) / run_dir_name(
        architecture, training_epsilon, seed
    )


# --- results trees: mirror the checkpoint trees above ---


def baseline_robustness_dir() -> Path:
    return RESULTS_ROOT / "robustness" / "baseline"


def static_adversarial_robustness_dir() -> Path:
    return RESULTS_ROOT / "robustness" / "adversarial" / "static_pgd"


def madry_adversarial_robustness_dir() -> Path:
    return RESULTS_ROOT / "robustness" / "adversarial" / "madry_pgd"


def fair_adversarial_robustness_dir(metric: str) -> Path:
    return RESULTS_ROOT / "robustness" / "fair_adversarial" / metric


def fair_adversarial_sweep_results_dir(metric: str) -> Path:
    return fair_adversarial_robustness_dir(metric) / "tau_sweep"


# --- generated datasets ---


def pgd_cache_path(
    training_style: str,
    architecture: str,
    seed: int,
    epsilon: float,
    tolerance: float | None = None,
) -> Path:
    """
    Cache location for the adversarial test set used to evaluate one model.

    Every axis that changes the model under attack must appear in this key, or one
    model gets evaluated against another model's adversarial examples. Tolerance is
    absent for training styles that have no tau (baseline, static, madry); caches
    written before the key carried a tau are a distinct filename and are never read.

    The key below still names every axis; the filename on disk is its SHA-256 hash,
    so distinct keys map to distinct (fixed-length, float-safe) filenames.
    """
    tolerance_key = "" if tolerance is None else f"tau{tolerance}_"
    key = f"{training_style}_{tolerance_key}{architecture}_{seed}_{epsilon}"
    filename = hashlib.sha256(key.encode()).hexdigest()
    return DATA_ROOT / "pgd_testing_data" / filename


def pgd_training_data_path(architecture: str, training_epsilon: float, seed: int) -> Path:
    """Pre-generated PGD training set consumed by scripts/train_static_adversarial.py."""
    return DATA_ROOT / "pgd_training_data" / f"{architecture}_eps{training_epsilon}_{seed}.pt"


def fgsm_cache_path(
    training_style: str, architecture: str, training_epsilon: float, seed: int
) -> Path:
    return (
        DATA_ROOT
        / "fgsm_training_data"
        / f"{training_style}_{architecture}_eps{training_epsilon}_{seed}.pt"
    )


# ----------------------------------


def results_tables_dir():
    return RESULTS_ROOT / "tables"
