# XRayPro Space-Group Screening (staging)

This is a clean, non-public staging tree for the core method used to rank molecular-crystal space groups from PXRD and molecular information.

The code exposes the following reader-facing workflows:

- train a PXRD + molecular-embedding classifier with an optional crystal-graph teacher;
- evaluate ordinary and class-balanced TOP-K recall;
- run graph-free inference from PXRD and molecular embeddings;
- preprocess experimental PXRD for classification;
- rank precomputed candidate PXRD profiles with constrained dynamic time warping (DTW), optionally after an energy filter.

The formal model uses a 3,400-point PXRD grid, a squeeze-and-excitation residual PXRD encoder, cached molecular embeddings, and a lightweight two-layer graph encoder during training. The graph pathway supplies classification and representation-alignment supervision. Evaluated inference logits use only PXRD and the molecular embedding.

## Scope

This repository is a code staging artifact, not a public release and not a complete paper-reproduction package. It intentionally excludes:

- licensed CSD and ICDD/PDF source records and row-level derivatives;
- molecular identifiers, experimental PXRD arrays, crystal coordinates and CIFs;
- trained checkpoints whose redistribution rights have not been cleared;
- manuscript, figure, Supporting Information and internal evidence builders;
- case-specific CSP generation, MACE relaxation, packing deduplication and PackingSimilarity validation.

Candidate generation and relaxation are external to this package. DTW is a PXRD-consistency ranking signal and does not establish crystal-packing identity.

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
them a second time. CIF simulation and licensed database extraction remain
external; they are not performed by this command. Use the same preparation and embedding protocol for training and
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
metrics, numerical definitions and the separate external CSP/DFT boundary.

Run the synthetic CPU regression suite with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*.py'
```

For reference DTW verification, install `.[verification]`. The optional test
compares both Python and C dtaidistance implementations. If the environment
reports an MKL OpenMP symbol error, prefix the command with
`MKL_THREADING_LAYER=GNU`; this is an environment workaround, not a model setting.

## Release status

The software license, citation metadata, authorship, public repository visibility, model-weight release and any data/Zenodo deposit remain pending separate approval. Do not publish this staging tree as-is.
