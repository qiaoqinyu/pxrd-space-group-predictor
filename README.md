# PXRD Space Group Predictor

## Unified CLI

For a new Linux x86-64 installation, follow the tested
[Conda installation steps and version locks](environments/README.md), then run
`pxrd-sg --help` and `pxrd-sg doctor`.
The `predict`, `run`, `rank`, and `doctor` commands provide a single interface to
the existing adapters. `run` requires an explicit candidate budget and GPU;
prediction alone never starts CSP. See [CLI guide](docs/cli.md) and
[environment profiles](environments/README.md). Download the paper's
[trained classifier](docs/model-weights.md) and supply a local ChemBERTa snapshot.
The weights are Release assets, separate from the source and wheel.
The full GPU workflow uses a separate BOMLIP/MACE Conda environment and requires
a compatible NVIDIA driver and CUDA toolkit. The tested installation uses no
container runtime.

This repository provides the core method used to rank molecular-crystal space groups from PXRD and molecular information.

The code exposes the following reader-facing workflows:

- train a PXRD + molecular-embedding classifier with an optional crystal-graph teacher;
- evaluate ordinary and class-balanced TOP-K recall;
- run graph-free inference from PXRD and molecular embeddings;
- preprocess experimental PXRD for classification;
- rank precomputed candidate PXRD profiles with constrained dynamic time warping (DTW), optionally after an energy filter;
- prepare conformers, run pinned BOMLIP-CSP generation and two-stage MACE relaxation, deduplicate structures and simulate PXRD;
- optionally validate crystal packing with a separately licensed CSD Python API. This step is unavailable without that license and local CCDC installation.

The formal model uses a 3,400-point PXRD grid, a squeeze-and-excitation residual PXRD encoder, cached molecular embeddings, and a lightweight two-layer graph encoder during training. The graph pathway supplies classification and representation-alignment supervision. Evaluated inference logits use only PXRD and the molecular embedding.

## Scope

This repository provides the core software, not a complete paper-reproduction package. It intentionally excludes:

- licensed CSD and ICDD/PDF source records and row-level derivatives;
- molecular identifiers, experimental PXRD arrays, crystal coordinates and CIFs;
- trained checkpoints whose redistribution rights have not been cleared;
- manuscript, figure, Supporting Information and internal evidence builders;
- historical case-specific calculation outputs and proprietary computational engines.

Downstream adapters call separately installed scientific engines. See [the downstream workflow](docs/downstream-workflow.md) for dependencies, fixed parameters and commands. DTW is a PXRD-consistency ranking signal and does not establish crystal-packing identity.

## Install

Python 3.10 or newer is required.

```bash
python -m pip install -e .
```

## Array contract

Training and evaluation use an NPZ bundle. Required inference arrays are:

```text
pxrd               float32 [N, L]
smiles_embedding   float32 [N, D]
```

Training additionally requires:

```text
labels              int64   [N]
atom_fea            float32 [N, A, F]
nbr_fea             float32 [N, A, K, E]
nbr_fea_idx         int64   [N, A, K]
atom_mask           bool    [N, A]
```

Evaluation requires `labels`; graph arrays are optional because reported prediction logits do not use the graph pathway. Optional `sample_id` values are emitted in prediction files.

All feature arrays must contain finite real values representable as float32.
Labels are zero-based class indices, not space-group numbers, and must be smaller
than the model's `num_classes`. Every split must use the same index mapping.
Graph neighbor indices must be integers in `[0, A)`. Masks must be boolean with
at least one real atom per sample; neighbors of real atoms must refer to real
atoms. Padded atom rows may use index zero. NaN, infinity and malformed graph
inputs are rejected rather than repaired.

The classifier consumes already prepared profiles. The formal grid is
`2theta[i] = 2.0 + 0.02 * i` degrees for `i = 0, ..., 3399`, ending at 69.98
degrees. Use `tools/prepare_pxrd.py` for experimental classification inputs: linear
interpolation with zero fill outside measured support, a centered 601-point
fifth-percentile baseline with nearest-edge padding, clipping at zero and
maximum scaling to 100. Outputs are linear intensities, not log1p caches.
Training and prediction consume these prepared arrays without preprocessing
them a second time. CIF simulation is available in the downstream workflow; licensed database extraction remains external. Neither is performed by this command. Use the same preparation and embedding protocol for training and
inference. Matching the array length alone does not establish compatibility
with a checkpoint. Custom grids are supported for models trained on those grids.

## Class definitions and fine-tuning

Set `num_classes` in the model JSON to preserve the full label space even when
a training subset omits classes. Without an explicit count, a class mapping or
an initial checkpoint, the count is inferred as `max(train_labels) + 1`.
Validation labels outside that range are rejected before training starts.

An optional `class_space_groups` list in the model JSON maps indices to unique
International Tables space-group numbers (1 through 230), with a single
`null` entry for an aggregate Others class. Others is displayed as `Others`,
never as a concrete CSP space group. For example,
`"num_classes": 3, "class_space_groups": [1, 2, 19]` maps class index 2 to
space group 19. This is a small format example. `configs/model.example.json` contains the
paper mapping: `[1, 2, 4, 7, 14, 18, 19, 29, 33, 61, null]`.
The list can also define the class count when `num_classes` is omitted.
The mapping is saved in the checkpoint. Evaluation and inference add mapped
space-group columns when it is present; otherwise outputs remain class indices.

With `--init-checkpoint`, training inherits the initial model configuration,
including the full class count and mapping. An optional `--config` must remain
compatible with that configuration. Thus a subset missing the highest-indexed
class can be fine-tuned without shrinking the classifier head.

## Commands

```bash
python tools/embed_smiles.py --input smiles.csv --output molecular_embeddings.npz
python tools/train.py --train data/train.npz --val data/val.npz --output-dir runs/example
python tools/evaluate.py --checkpoint runs/example/best.pt --data data/test.npz --output-dir evaluation
python tools/infer.py --checkpoint runs/example/best.pt --data data/inference.npz --output predictions.csv
python tools/score_pxrd_candidates.py --reference observed.npy --candidates candidates.npy --output scores.csv
```

`embed_smiles.py` defaults to `DeepChem/ChemBERTa-100M-MLM` revision
`f5c45f44d3061f0346888f5c09db17ec1146d29d`, attention-mask mean pooling and a
maximum token length of 192. Supply canonical isomeric SMILES
prepared with the same canonicalization protocol (RDKit 2025.09.1 in the paper);
this tool does not canonicalize text or reconstruct the licensed molecule splits. Install the optional embedding dependencies with
`python -m pip install -e '.[embedding]'`.

A synthetic, non-scientific smoke workflow is provided in `examples/minimal_workflow.py`. Its values only verify software contracts.

The DTW command requires reference and candidate arrays on the same uniform
grid and support. Use `--start-index` and `--stop-index` for a shared contiguous
crop; do not drop isolated missing points and thereby compress the angular
axis. Non-finite profiles and profiles with no positive signal after
preprocessing are rejected. Negative values are clipped before reference-only minimum subtraction. Both
inputs then undergo max100 scaling, five-point max pooling and independent
L2 normalization. Intensities remain linear: there is no square-root transform.
The local cost is absolute difference (scalar Euclidean distance), and the
accumulated path cost is divided by the square root of the pooled length.
The default radius 30 equals `dtaidistance` window 31, with zero penalty and
no endpoint relaxation. Only the shared measured support is scored. These DTW transformations are separate from classifier
preprocessing. `--band-points` is measured on the pooled grid, and incomplete
trailing pooling bins are discarded.

Checkpoint loading uses PyTorch's restricted `weights_only=True` loader.
Existing staging schema-1 checkpoints without a class mapping remain readable;
arbitrary pickled Python objects are not supported. Training rejects non-finite
losses and gradients; evaluation and inference reject non-finite logits before
writing prediction outputs.

Use `--energies energies.npy --candidate-ids ids.npy` with the DTW command
to retain the 200 lowest-energy candidates and mark the ten best DTW scores.
Input candidates must already be deduplicated within space groups and pooled
over the selected groups for one molecule. Energy ties use string candidate IDs;
DTW ties use energy, then ID. Without these files, the command only scores all
supplied profiles; it does not apply the paper screening sequence. No packing
labels or experimental structure are used for ranking.

See [paper-to-code correspondence](docs/paper-code-map.md) for training stages,
metrics, numerical definitions, downstream adapters and the external DFT boundary.

Run the synthetic CPU regression suite with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*.py'
```

For reference DTW verification, install `.[verification]`. The optional test
compares both Python and C dtaidistance implementations. If the environment
reports an MKL OpenMP symbol error, prefix the command with
`MKL_THREADING_LAYER=GNU`; this is an environment workaround, not a model setting.

## Agent skill

The repository includes [PXRD space-group screening](skills/pxrd-space-group-screening/SKILL.md),
a reusable skill for agents working with this project. Clone the repository and
install its dependencies in a Python 3.10+ environment:

```bash
git clone https://github.com/qiaoqinyu/pxrd-space-group-predictor.git
cd pxrd-space-group-predictor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[embedding]'
```

Open this checkout in an agent that supports project skills. The symlink under
`.agents/skills/` points to the canonical skill. In Codex, invoke
`$pxrd-space-group-screening`; with other agents, provide the full path to
`skills/pxrd-space-group-screening/SKILL.md` and ask the agent to follow it.
Keep the skill with the repository because it calls the repository's CLI tools.
No global skill installation is required.

For example, ask your agent:

> Use the pxrd-space-group-screening skill to predict the top three space groups
> from my observed.txt and molecules.csv, using my compatible trained checkpoint.
> Save the commands, input hashes and predictions in a new runs/ directory.

**The paper's full_pretrained checkpoint for seed 240427 is publicly available.**
The [model guide](docs/model-weights.md) provides the download, SHA256, class
mapping and preprocessing/embedding requirements. This is one trained model,
not an ensemble or all five runs used for aggregate paper results. The agent must
verify the downloaded checkpoint before prediction. Candidate DTW ranking can
run without a classifier checkpoint. The synthetic example tests execution only.

The skill also guides conformer preparation, BOMLIP-CSP generation, two-stage
MACE relaxation, deduplication, simulated PXRD and optional CCDC packing
validation. See [the downstream workflow](docs/downstream-workflow.md) for
commands, dependencies and verification limits. DFT remains a separate workflow.
The wheel includes the CLI adapters and installs the skill, guides, configs,
example generator and environment profiles under
`<environment>/share/pxrd-space-group-predictor/`.

To assemble prepared profiles and cached molecular embeddings, use:

```bash
python tools/build_inference_bundle.py --manifest profiles.csv --embeddings embeddings.npz --config configs/model.example.json --output runs/task/inference.npz
```

The CSV requires `sample_id,pxrd_path`; relative paths resolve against the CSV.
The NPZ requires string `sample_id` and numeric `smiles_embedding` arrays.
Rows are joined by ID, and the ID sets must match exactly. Duplicate IDs,
invalid arrays and existing output paths are rejected.

For a trusted original SE-Res/simple-graph embedding checkpoint, use
`tools/import_legacy_checkpoint.py --help`. Supply a verified SHA256, the verified
model configuration and an explicit provenance label. The importer requires the
3400-point grid and creates a separate schema-1 copy by renaming graph keys;
it does not change learned tensors. Check original-versus-converted predictions
before scientific use. The designated public checkpoint is described in the
[model guide](docs/model-weights.md). Licensed inputs and other unpublished
research assets remain excluded.

## Release status

This repository is publicly maintained at https://github.com/qiaoqinyu/pxrd-space-group-predictor. Transfer to the pic-ai-robotic-chemistry organization is planned only after the author requests it. The designated classifier is available in the `weights-v1` Release. Software/model licensing, citation metadata and data/Zenodo deposits remain pending. Public visibility does not grant an open-source license.
