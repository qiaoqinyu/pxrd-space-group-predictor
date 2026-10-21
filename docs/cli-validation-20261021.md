# Unified CLI validation — 21 October 2026

The `0.2.0.dev1` interface wraps the existing scientific adapters. The checks
below establish software execution within the stated environments. They do not
reproduce the paper's full case study or establish prediction accuracy.

## Checks completed

- The full unit suite passed 58 tests with no skips on Linux/Python 3.10.
- A new venv with system-site-packages disabled installed the CPU prediction
  dependencies. `pip check`, `pxrd-sg doctor` and real-checkpoint prediction passed.
  Its complete resolved package versions are in
  `environments/predict-cpu-linux.lock.txt`.
- A wheel was installed and invoked from outside the source checkout. It includes
  the adapters, skill, example generator, protocol files and environment profiles.
- On an NVIDIA A800, raw synthetic PXRD and ethanol SMILES passed through fresh
  ChemBERTa embedding, real classifier inference, SG1 generation, two-stage MACE,
  within-group deduplication, simulated PXRD, energy/DTW scoring and CIF export.
  The explicit budget was one candidate. The SG1 override is a test setting,
  not a prediction or a claim about the synthetic pattern's symmetry.
- Repeating the command with `--resume` reused all eight verified stages.
  Independent `rank` execution used the relaxed manifest without a classifier.
- A regression test demonstrated that the old supervisor left a nested engine
  alive after timeout. The corrected supervisor terminates that process, and the
  test passes. Identity changes, altered outputs, overlapping runs and an Others
  shortlist are also covered.

## Boundaries

The A800 prediction venv inherits an existing scientific environment; the Linux
CPU prediction test uses a clean venv. The MACE environment is an existing working
engine, not a newly installed or containerized engine. Its dependencies differ
from those used for prediction, so the CLI accepts `--engine-python`.

The prediction-only Docker recipe has not been built: the validation host denies
access to its Docker daemon. No complete GPU image is published. macOS and native
Windows installations were not tested; WSL2/Linux is the full-workflow target.

At the time of these CLI checks, the classifier was supplied privately. The
subsequent `weights-v1` Release makes that same checkpoint publicly available;
see [model weights](model-weights.md). No restricted data or historical case
outputs are included in that release. A successful small run does not
certify force convergence, structural identity or reproducibility across GPU
architectures. Optional CCDC validation remains outside these checks.
