# Reproducibility Artifacts

This directory preserves the small, text-based experiment metadata needed to
interpret the final result summaries. The original `trained_models/` hierarchy
is nested below this directory so architecture, training method, seed, training
epsilon, and fairness tolerance remain visible in each path.

The integration also preserves final CSV summaries, named paper-facing figures,
per-seed baseline reports and diagnostic plots, and small named PDF/TeX reports
under `results/`. `ARTIFACT_MANIFEST.tsv` records the original ZIP path, integrated
path, size, SHA-256, inferred experiment association, and paper-reference status.

Intentionally excluded from Git are model checkpoints, `*.pt`, `*.pth`, and
`*.ckpt` files; raw/downloaded data; FGSM/PGD attack inputs and caches; training
caches; Python/editor/OS caches; environments; and LaTeX auxiliary files. Those
excluded artifacts remain available in the original student archive:

- Path: `/home/rivas/research/summer-2026/jfrank_fair-safe-opt.zip`
- SHA-256: `7cb6fbd39b93a92275b4c943ea678a3c53cbc1ec78834cda2534a701a7457924`

Result CSVs map to the evaluation entrypoints under `scripts/robustness_eval/`.
Paper-facing figures map to the readers under `scripts/plotting/`, which consume
the committed per-seed decimal CSVs and perform seed aggregation for display.
Training configurations and histories map to the training entrypoints under
`scripts/training/`. No experiment was rerun during integration.
