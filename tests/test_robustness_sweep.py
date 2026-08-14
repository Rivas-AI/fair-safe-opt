import torch
import pytest

import fair_safe_opt.evaluation.robustness as robustness
from fair_safe_opt.evaluation.robustness import (
    evaluation_sweep,
    summarize_evaluation_sweep,
    tolerance_sweep_evaluation,
)
from fair_safe_opt.utils.paths import (
    fair_adversarial_root,
    fair_adversarial_run_dir,
    pgd_cache_path,
    run_dir_name,
)

_ARCHITECTURES = ["logistic", "small_mlp"]
_EPSILONS = [0.0, 0.1]
_SEEDS = [0, 1]
_TOLERANCES = [0.005, 0.05]
_TRAINING_STYLE = "dp_fair"
_METRIC = "dp"


class FakeSplit:
    # Group 0 is predicted perfectly, group 1 entirely wrong (same tensors as
    # test_metric_table.py), so the metric tables have known exact values.
    x_test = torch.zeros(8, 3)
    y_test = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    a_test = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    continuous_mask = torch.tensor([True, True, True])
    feature_min = torch.zeros(3)
    feature_max = torch.ones(3)


class FakeModel:
    def load_state_dict(self, state_dict) -> None:
        pass

    def preds_probs(self, features, threshold):
        preds = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        probs = torch.tensor([0.9, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1, 0.9])
        return preds, probs


class FakePerfectModel(FakeModel):
    # Predicts every label correctly, so attack_success_rate has a nonzero
    # denominator in every group (it is NaN for a group with no correct
    # baseline predictions).
    def preds_probs(self, features, threshold):
        preds = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        probs = torch.tensor([0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
        return preds, probs


def make_checkpoints(models_dir, tolerance=None) -> None:
    # 0.0 is evaluation_sweep's default; none of these tests pass a training_epsilon.
    training_epsilon = 0.0
    tolerance_dir = models_dir if tolerance is None else models_dir / f"tau_{tolerance}"
    for architecture in _ARCHITECTURES:
        for seed in _SEEDS:
            save_dir = tolerance_dir / run_dir_name(architecture, training_epsilon, seed)
            save_dir.mkdir(parents=True)
            torch.save({}, save_dir / "best_model.pt")


def patch_heavy_dependencies(monkeypatch, tmp_path, model_cls=FakeModel) -> None:
    monkeypatch.setattr(robustness, "load_adult", lambda random_state: FakeSplit())
    monkeypatch.setattr(robustness, "build_model", lambda *args, **kwargs: model_cls())
    # Identity attack: adversarial features equal clean features.
    monkeypatch.setattr(robustness, "pgd_attack", lambda model, x, y, **kwargs: x)
    # Redirect the cache into tmp_path, keeping the real key format.
    monkeypatch.setattr(
        robustness,
        "pgd_cache_path",
        lambda *args, **kwargs: tmp_path / "pgd_data" / pgd_cache_path(*args, **kwargs).name,
    )
    (tmp_path / "pgd_data").mkdir()


def patch_checkpoint_root(monkeypatch, tmp_path) -> None:
    # tolerance_sweep_evaluation builds its own checkpoint root per tolerance;
    # point that at the tree make_checkpoints() writes.
    monkeypatch.setattr(
        robustness, "fair_adversarial_root", lambda metric, tolerance: tmp_path / f"tau_{tolerance}"
    )


def test_evaluation_sweep_returns_group_tables(monkeypatch, tmp_path) -> None:
    patch_heavy_dependencies(monkeypatch, tmp_path)
    make_checkpoints(tmp_path)

    tables = evaluation_sweep(
        _ARCHITECTURES, _EPSILONS, _SEEDS, _TRAINING_STYLE, model_save_dir=tmp_path
    )

    assert set(tables) == {"combined", "group_0", "group_1"}
    for table in tables.values():
        assert len(table) == len(_ARCHITECTURES) * len(_EPSILONS) * len(_SEEDS)
        assert "group" not in table.columns
        assert list(table["seed"].unique()) == _SEEDS

    # The persisted contract retains per-seed decimal metrics.
    combined = tables["combined"].set_index(["architecture", "epsilon", "seed"])
    assert combined.loc[("logistic", 0.0, 0), "balanced_accuracy"] == 0.5
    assert combined.loc[("logistic", 0.0, 0), "auc"] == 0.5


def test_summarize_evaluation_sweep_averages_and_scores(monkeypatch, tmp_path) -> None:
    patch_heavy_dependencies(monkeypatch, tmp_path)
    make_checkpoints(tmp_path)
    raw_tables = evaluation_sweep(
        _ARCHITECTURES, _EPSILONS, _SEEDS, _TRAINING_STYLE, model_save_dir=tmp_path
    )
    tables = summarize_evaluation_sweep(raw_tables)

    for table in tables.values():
        assert len(table) == len(_ARCHITECTURES) * len(_EPSILONS)
        assert "seed" not in table.columns

    combined = tables["combined"].set_index(["architecture", "epsilon"])
    assert combined.loc[("logistic", 0.0), "balanced_accuracy"] == 50.0
    assert combined.loc[("logistic", 0.0), "auc"] == 50.0
    assert (
        tables["group_0"]
        .set_index(["architecture", "epsilon"])
        .loc[("logistic", 0.0), "balanced_accuracy"]
        == 100.0
    )
    assert (
        tables["group_1"]
        .set_index(["architecture", "epsilon"])
        .loc[("logistic", 0.0), "balanced_accuracy"]
        == 0.0
    )


def test_evaluation_sweep_deltas_are_zero_when_attack_changes_nothing(
    monkeypatch, tmp_path
) -> None:
    # With an identity attack the adversarial metrics equal the clean metrics,
    # so every epsilon > 0 delta must be exactly zero for every group.
    patch_heavy_dependencies(monkeypatch, tmp_path, model_cls=FakePerfectModel)
    make_checkpoints(tmp_path)

    raw_tables = evaluation_sweep(
        _ARCHITECTURES, _EPSILONS, _SEEDS, _TRAINING_STYLE, model_save_dir=tmp_path
    )
    tables = summarize_evaluation_sweep(raw_tables, as_deltas=True)

    for name, table in tables.items():
        metric_cols = [col for col in table.columns if col not in ["architecture", "epsilon"]]
        deltas = table[table["epsilon"] > 0][metric_cols]
        assert (deltas == 0.0).all().all(), f"nonzero deltas in {name}:\n{table}"

    # Baseline rows keep their raw (scored) values.
    combined = tables["combined"].set_index(["architecture", "epsilon"])
    assert combined.loc[("logistic", 0.0), "balanced_accuracy"] == 100.0


def test_evaluation_sweep_missing_checkpoint_raises(monkeypatch, tmp_path) -> None:
    patch_heavy_dependencies(monkeypatch, tmp_path)

    with pytest.raises(FileNotFoundError, match="No trained static-adversarial model"):
        evaluation_sweep(
            _ARCHITECTURES, _EPSILONS, _SEEDS, _TRAINING_STYLE, model_save_dir=tmp_path / "empty"
        )


def test_writer_and_reader_agree_on_checkpoint_path() -> None:
    # scripts/train_fair_robust.py writes via fair_adversarial_run_dir; evaluation_sweep
    # reads via fair_adversarial_root / run_dir_name. If these two ever disagree, every
    # tau evaluates the wrong checkpoint.
    written = fair_adversarial_run_dir("dp", 0.005, "logistic", 0.3, 0)
    read = fair_adversarial_root("dp", 0.005) / run_dir_name("logistic", 0.3, 0)

    assert written == read
    assert "tau_0.005" in written.parts


def test_evaluation_sweep_finds_checkpoint_under_tolerance_scoped_root(
    monkeypatch, tmp_path
) -> None:
    # Callers pass an already tolerance-scoped root, so a root built for one tau
    # cannot resolve another tau's models.
    patch_heavy_dependencies(monkeypatch, tmp_path)
    make_checkpoints(tmp_path, tolerance=0.005)

    evaluation_sweep(
        _ARCHITECTURES,
        _EPSILONS,
        _SEEDS,
        _TRAINING_STYLE,
        model_save_dir=tmp_path / "tau_0.005",
        tolerance=0.005,
    )

    with pytest.raises(FileNotFoundError):
        evaluation_sweep(
            _ARCHITECTURES,
            _EPSILONS,
            _SEEDS,
            _TRAINING_STYLE,
            model_save_dir=tmp_path / "tau_0.05",
            tolerance=0.05,
        )


def test_each_tolerance_writes_its_own_attack_cache(monkeypatch, tmp_path) -> None:
    # The bug this guards against: reusing one tolerance's cached adversarial
    # examples to evaluate a different tolerance's model.
    patch_heavy_dependencies(monkeypatch, tmp_path)
    patch_checkpoint_root(monkeypatch, tmp_path)
    for tolerance in _TOLERANCES:
        make_checkpoints(tmp_path, tolerance=tolerance)

    tolerance_sweep_evaluation(
        _ARCHITECTURES, _EPSILONS, _SEEDS, _TOLERANCES, _METRIC, _TRAINING_STYLE
    )

    cache_files = sorted(path.name for path in (tmp_path / "pgd_data").iterdir())
    assert len(cache_files) == len(_TOLERANCES) * len(_ARCHITECTURES) * len(_EPSILONS) * len(_SEEDS)
    # Filenames are hashes of the cache key, so match on the name the key produces
    # rather than a readable tau substring.
    for tolerance in _TOLERANCES:
        expected = pgd_cache_path(
            _TRAINING_STYLE, _ARCHITECTURES[0], _SEEDS[0], _EPSILONS[0], tolerance
        ).name
        assert expected in cache_files


def test_tolerance_sweep_does_not_read_old_style_cache(monkeypatch, tmp_path) -> None:
    # Caches keyed without a tolerance are ambiguous; the attack must be rerun
    # rather than falling back to them.
    patch_heavy_dependencies(monkeypatch, tmp_path)
    make_checkpoints(tmp_path, tolerance=0.005)

    attack_calls = []
    monkeypatch.setattr(
        robustness, "pgd_attack", lambda model, x, y, **kwargs: attack_calls.append(1) or x
    )

    for architecture in _ARCHITECTURES:
        for seed in _SEEDS:
            for epsilon in _EPSILONS:
                torch.save(
                    FakeSplit.x_test,
                    tmp_path / "pgd_data" / f"{_TRAINING_STYLE}_{architecture}_{seed}_{epsilon}",
                )

    evaluation_sweep(
        _ARCHITECTURES,
        _EPSILONS,
        _SEEDS,
        _TRAINING_STYLE,
        model_save_dir=tmp_path / "tau_0.005",
        tolerance=0.005,
    )

    assert len(attack_calls) == len(_ARCHITECTURES) * len(_EPSILONS) * len(_SEEDS)


def test_tolerance_sweep_evaluation_tags_rows_with_tolerance(monkeypatch, tmp_path) -> None:
    patch_heavy_dependencies(monkeypatch, tmp_path)
    patch_checkpoint_root(monkeypatch, tmp_path)
    for tolerance in _TOLERANCES:
        make_checkpoints(tmp_path, tolerance=tolerance)

    tables = tolerance_sweep_evaluation(
        _ARCHITECTURES, _EPSILONS, _SEEDS, _TOLERANCES, _METRIC, _TRAINING_STYLE
    )

    assert set(tables) == {"combined", "group_0", "group_1"}
    for table in tables.values():
        assert len(table) == (len(_TOLERANCES) * len(_ARCHITECTURES) * len(_EPSILONS) * len(_SEEDS))
        assert list(table["tolerance"].unique()) == _TOLERANCES
        assert list(table["seed"].unique()) == _SEEDS

    summarized = summarize_evaluation_sweep(tables)
    for table in summarized.values():
        assert len(table) == len(_TOLERANCES) * len(_ARCHITECTURES) * len(_EPSILONS)
        assert "seed" not in table.columns

    combined = summarized["combined"].set_index(["tolerance", "architecture", "epsilon"])
    assert combined.loc[(_TOLERANCES[0], "logistic", 0.0), "balanced_accuracy"] == 50.0


def test_config_has_keys_used_by_eval_script() -> None:
    # scripts/adv_robustness_eval.py and evaluation_sweep read these directly.
    for key in [
        "architecture_sweep",
        "epsilon_sweep",
        "seed_sweep",
        "training_epsilon",
        "steps",
        "threshold",
        "hidden_dim",
        "dropout",
        "tolerance_sweep",
    ]:
        assert key in robustness.config, f"training_configs.yaml is missing '{key}'"
