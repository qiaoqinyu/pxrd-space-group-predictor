# Staging audit record

Date: 2026-09-09

## Post-review corrections (2026-09-09)

The original snapshot described below preceded the input-contract fixes. The
current tree adds `tests/test_input_contracts.py` and revises the runtime,
training and prediction interfaces without changing the encoder topology.

- Reject non-finite feature arrays, logits, losses and gradients, out-of-range
  labels, malformed graph tensors and invalid neighbor indices.
- Reject non-finite or signal-free DTW profiles before candidate ranking.
- Keep explicit class counts for subsets and inherit the initial checkpoint's
  configuration during fine-tuning. Validate validation-set dimensions and
  labels before training starts.
- Save optional `class_space_groups` in the model configuration and emit mapped
  prediction columns. No paper-specific mapping is guessed or bundled.
- Load schema-1 checkpoints with `weights_only=True`, including legacy configs
  without a mapping. Custom pickled Python objects are rejected.
- Expand the synthetic CPU suite to 16 tests covering these regressions,
  checkpoint round trips with optimizer state, full training, subset
  fine-tuning, graph-free evaluation/inference and DTW output.

CPU verification uses the documented `academic` environment. Formal GPU runs,
real molecular embedding verification and scientific reproduction remain
outside this correction pass. The prior source repository was not edited.

## Scope

This audit covers the clean code staging tree only. It does not authorize publication, data redistribution, model-weight release or a repository visibility change.

## Tree

- 30 reader-facing source, documentation and configuration files after this audit record was added.
- Approximately 1,500 lines of Python across `src/`, `tools/`, `examples/` and `tests/`.
- No Git repository was initialized; public history review therefore remains pending.
- No license or `CITATION.cff` was created because authorship, license scope and release approval remain unresolved.

## Verified behavior

- `pyproject.toml` parsed successfully with Python 3.13 `tomllib`.
- All Python sources compiled successfully in the project's `academic` Conda environment (Python 3.10.18).
- Four `unittest` smoke checks passed:
  - graph-free inference logits equal prediction-path logits when graph tensors are present;
  - formal graph classification and Barlow-style alignment loss is finite;
  - PXRD-only and molecular-embedding-only controls require only their selected input;
  - an identical PXRD profile receives the minimum constrained-DTW distance.
- Help checks passed for `train.py`, `evaluate.py`, `infer.py`, `score_pxrd_candidates.py`, `embed_smiles.py` and `make_synthetic_data.py`.
- A strict synthetic end-to-end run completed:
  - generated train, validation, labeled graph-free test and graph-free inference bundles;
  - trained one CPU epoch and wrote `best.pt` plus metrics;
  - evaluated the checkpoint and wrote metrics plus predictions;
  - ran graph-free inference;
  - ranked candidate PXRD profiles with constrained DTW;
  - verified every expected output was non-empty.
- A separate one-epoch SMILES-only training run completed.
- A separate no-graph-supervision run completed from a bundle without graph arrays, with graph classification and alignment weights both resolved to zero.

## Architecture equivalence

The staging model was compared with the private formal SERes/simple model configuration:

- parameters: 4,833,230 in both implementations;
- parameter tensors: 122 in both implementations;
- state tensors: 182 in both implementations;
- ordered parameter shapes: identical;
- ordered state-tensor shapes: identical.

The public API changes the reader-facing class and module names and permits graph-free inference. It does not change the formal prediction-pathway topology.

## Static boundary scan

The staged source tree returned:

- legacy internal experiment name hits: 0;
- absolute local/server path hits: 0;
- credential-pattern hits: 0;
- restricted/result file extensions (`.cif`, `.xy`, `.npy`, `.npz`, `.pt`, `.pth`, `.ckpt`, `.pkl`, `.csv`): 0 tracked files.

The tree contains no imports from private evidence modules, manuscript assets, result artifacts or the external CSP repository.

## Remaining gates

- The pinned ChemBERTa command passed compile and CLI checks, but model download and numerical embedding verification were not run in this CPU audit.
- Full GPU training, formal checkpoint compatibility and scientific-result reproduction were not run; the private evidence package remains authoritative for reported results.
- Generic graph and experimental-PXRD preparation from raw licensed records is intentionally absent pending rights and provenance decisions.
- License, authorship, `CITATION.cff`, public repository creation, Git history review, model weights, data assets, Zenodo and DOI remain pending separate approval.
- Advisor review of the staging scope and paper-facing wording remains pending.
