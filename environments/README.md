# Environment profiles

The CLI runs prediction/processing in one environment and can launch generation
and relaxation with `--engine-python` in another. Neither requires CSD for the
public workflow. Do not copy the whole research environment or upgrade its engine
dependencies by installing the predictor into it.

## Prediction and ranking

For CPU prediction/ranking on Linux x86-64, use a dedicated Python 3.10
environment and the complete resolved package-version list:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.7.1+cpu
.venv/bin/python -m pip install -r environments/predict-cpu-linux.lock.txt
.venv/bin/python -m pip install --no-deps .
.venv/bin/pxrd-sg doctor
.venv/bin/python -m pip check
```

The CPU lock was captured from a new venv with system-site-packages disabled.
It pins all installed runtime package versions, including transitive dependencies;
it does not pin wheel hashes or operating-system libraries. Install PyTorch first
as shown because its CPU wheel uses a separate index. `predict-linux.txt` is the
broader direct-dependency profile for an existing compatible GPU environment.
Neither is the historical training environment. Preserve the package versions
in `run.json` when interpreting results across machines. Linux x86-64 validation
does not establish that this lock works on Apple Silicon or native Windows.

## MACE engine

Follow the installation instructions for the clean BOMLIP commit in
`docs/downstream-workflow.md`. The verified engine has Python 3.10, PyTorch 2.4.1,
NumPy 1.26.4, SciPy 1.14.1, pymatgen 2024.11.13, ASE 3.23.0, MACE 0.3.13,
cuEquivariance 0.4.0 and Ninja 1.13.2. Versions alone do not identify patched
sources; the adapter uses the pinned BOMLIP source snapshot and MACE model hash.
This is a compatibility record, not a standalone engine installer.

The unified CLI supplies its installed adapter path to the engine interpreter.
Use `doctor --full` with the engine and model paths before a calculation.

## Container recipe

`containers/Dockerfile.predict` is an optional prediction/ranking image recipe.
Models and data are mounted at runtime and are not included. It is not a full
CUDA/MACE image. A complete GPU image requires an independently tested engine
installation. The current validation host denies access to its Docker daemon;
this recipe has not been build-tested there.

```bash
docker build -f containers/Dockerfile.predict -t pxrd-sg:predict .
docker run --rm pxrd-sg:predict --help
```

Use absolute mounted paths and preserve the output directory for resume. The
host still needs compatible drivers/runtime for any future GPU image. Container
packaging does not waive third-party software or model licensing requirements.
