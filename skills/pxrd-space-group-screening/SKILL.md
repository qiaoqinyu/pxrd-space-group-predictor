---
name: pxrd-space-group-screening
description: Rank molecular-crystal space groups from experimental PXRD and SMILES, prepare and relax crystal candidates with BOMLIP-CSP/MACE, and rank or validate candidates through the project's CLI tools.
---

# PXRD-guided space-group screening

## Install and check the environment first

For a new Linux x86-64 installation, read `environments/README.md` before
installing dependencies. Prefer its tested Conda profiles and resolved version
locks: `predict-cpu-linux.yml` for prediction and ranking, plus
`mace-engine-linux.yml` for GPU generation and relaxation. Follow the complete
installation commands in that guide; the YAML files alone only create Python
and pip environments. Keep the two environments separate and pass the engine
interpreter with `--engine-python`. The GPU profile also requires a compatible
NVIDIA driver and CUDA toolkit. These profiles were validated on Linux x86-64;
other platforms require separate verification.

Reuse an existing dedicated installation after checking it with `pxrd-sg doctor`.
Before a full run, obtain the model assets from `docs/model-weights.md` and
`docs/downstream-workflow.md`, then run `pxrd-sg doctor --full` with the selected
GPU, engine interpreter, BOMLIP checkout and MACE model. Proceed only when the
checks pass. Use explicit input, model and output paths; local paths from another
chat are not portable installation settings.

## Preferred installed interface

Use `pxrd-sg --help` and read `docs/cli.md` when the installed CLI is available.
For a wheel installation, resolve the documentation and configs from
`<environment>/share/pxrd-space-group-predictor/`; this skill is under its
`skills/` directory. The wheel does not create a source checkout.
Run `pxrd-sg doctor` to check the environment. `predict` accepts raw PXRD, SMILES,
the measured wavelength, a trusted checkpoint/hash and a local ChemBERTa snapshot.
`run` adds candidate generation, MACE and ranking with explicit budget and GPU
selection. `rank` accepts raw PXRD and a relaxed-candidate manifest without a
classifier. Preserve `run.json`, logs and numbered stage directories. Reuse
completed stages only with the identical command plus `--resume`.

If the predicted shortlist contains Others, full execution stops rather than
inventing a search group. Obtain concrete search groups before a new run. The
engine may use a separate Python environment. CCDC validation remains optional
and separately licensed. Download the classifier separately using
`docs/model-weights.md`; the wheel does not bundle weights. The direct adapters
below remain supported for individual stages.

The direct adapter examples below assume a source checkout. In that case, resolve any symlink to this skill directory first, then resolve the repository root from that directory (`../..`). Run commands from that root in its configured Python environment; inspect `README.md`, `pyproject.toml` and each tool's `--help` when needed. For an installed wheel, use `python -m tools.<name>` instead of `python tools/<name>.py`, with explicit paths to the bundled configs.

## Establish inputs and model provenance

For prediction, obtain experimental two-column 2theta/intensity text, canonical isomeric SMILES, stable sample IDs and a compatible trained checkpoint. Confirm that angles are degrees 2theta and record the radiation/wavelength used to obtain each pattern. The paper protocol uses Cu Kα radiation; different experimental conditions require a compatibility assessment, not silent conversion. Read `configs/paper_protocol.json` and the README's array contract. Synthetic examples validate execution only; they are not trained paper models.

The `weights-v1` GitHub Release provides the paper's full_pretrained checkpoint for seed 240427. Follow `docs/model-weights.md` to download it and verify its SHA256. Record the model configuration, class mapping and training provenance. Obtain ChemBERTa separately at the documented revision. If a required asset is unavailable, report it and stop prediction; candidate ranking can still run independently. Never replace a missing trained checkpoint with random weights. The public checkpoint is already in schema-1 format and does not need conversion. For other trusted historical SE-Res/simple-graph checkpoints, `tools/import_legacy_checkpoint.py` creates a separate schema-1 copy through exact graph-key renaming. Confirm prediction parity against the original implementation before scientific use. A single checkpoint does not reproduce aggregate results from multiple training runs.

## Predict space groups

Use a new output directory. The commands below illustrate one sample; replace paths with the user's files and run preparation once per sample.

```bash
python tools/prepare_pxrd.py --input observed.txt --output runs/task/prepared.npy
python tools/embed_smiles.py --input molecules.csv --output runs/task/embeddings.npz --device cpu
python tools/build_inference_bundle.py --manifest runs/task/profiles.csv --embeddings runs/task/embeddings.npz --config configs/model.example.json --output runs/task/inference.npz
python tools/infer.py --checkpoint model.pt --data runs/task/inference.npz --output runs/task/predictions.csv --device cpu
```

`molecules.csv` contains `sample_id,smiles`. `profiles.csv` contains `sample_id,pxrd_path`; paths are relative to that CSV, for example `sample-1,prepared.npy`. The bundle builder joins IDs and rejects duplicate, missing or extra IDs. Use the checkpoint's actual configuration instead of the example when it differs. `embed_smiles.py` encodes text without canonicalizing it; use the checkpoint's canonicalization and ChemBERTa revision/pooling protocol. Install the optional `embedding` dependencies only when needed and authorized. Existing compatible cached embeddings can replace the encoding step if their sample IDs and provenance are verified.

Classification preparation performs interpolation, local-percentile background subtraction and max100 scaling. Reuse already prepared compatible arrays directly; do not preprocess them twice. Verify one output row per input ID, finite probabilities and a known class mapping. Report Others as an aggregate class, never as a concrete space group. TOP-3 is a ranked shortlist, not proof that the experimental space group is included.

## Rank existing candidates (optional)

Obtain the raw experimental profile and simulated candidate profiles on the same uniform angular grid, with the measured interval identified. Classification-prepared arrays have a different baseline treatment and must not be reused as raw DTW inputs. Crop all arrays to the shared measured support using `--start-index` and exclusive `--stop-index`; the tool then subtracts the experimental minimum within that interval. Preserve gaps and the angular grid.

```bash
python tools/score_pxrd_candidates.py --reference observed_raw.npy --candidates candidates_raw.npy --energies energies.npy --candidate-ids ids.npy --start-index 0 --stop-index 3400 --output runs/task/scores.csv
```

Set crop indices from actual support; the example assumes the full 3400-point grid was measured. Candidates must already be deduplicated within space groups and pooled over the selected groups for one molecule. Energies must be comparable per molecule and IDs must follow candidate rows. With energies and IDs the defaults retain up to 200 lowest-energy candidates and select up to ten by DTW. Without these inputs the command scores all supplied candidates and does not apply energy preselection. Read `README.md` for five-point pooling, L2 normalization, the DTW band and tie rules. Verify output IDs, counts and finite scores. Do not use known packing labels or inject a reference structure during ranking.

## Generate, relax and process candidates (optional)

Read `docs/downstream-workflow.md` before this branch. Obtain explicit authorization for generation/GPU computation and confirm concrete search groups; do not launch a 5,000-candidate search from a prediction-only request. Use a clean BOMLIP-CSP checkout at the documented commit, its configured environment and a hash-verified MACE-OFF23-small model. Do not edit an existing upstream worktree, download substitute weights or use Others as a search group. Check available hardware and use a new output directory for each stage.

Run `tools/generate_crystal_candidates.py` with the requested single-component SMILES, stable sample ID and groups. The default is 5,000 candidates per group; smaller budgets are execution tests. Check `candidates.csv` and the hash-linked `generation.json`. Run `tools/relax_crystal_candidates.py` with that manifest and explicit model path. Molecular normalization comes from generation metadata, not a guessed cell divisor. Preserve candidate IDs and report step-limit statuses; completed output is not independent force-convergence certification.

Run `tools/process_crystal_candidates.py` on `relaxed.csv` to deduplicate within groups and simulate PXRD. Use its aligned candidate, energy and ID arrays with the ranking command above. Always use the raw experimental pattern and actual measured support for DTW. Inspect counts, hashes, logs and finite energies/scores at each boundary; fail or report unresolved errors instead of silently omitting candidates.

For optional post-ranking reference validation, run `tools/validate_crystal_packing.py` only when the operator has a separate CSD license and a compatible CCDC Python API environment. The public repository does not include CSD records or the CCDC API. Users without this license must skip this step. Use an unoptimized experimental reference and a manifest for the intended candidate set. Errors/timeouts are unresolved, never negative matches. Reference labels cannot influence ranking. DFT remains separate.

## Deliver a traceable result

Save commands, code revision, input/model hashes, preprocessing and embedding settings, class mapping, measured support, output paths and checks in the task output directory. Respect the redistribution terms for supplied data and checkpoints; share only assets authorized for release. Distinguish software execution checks from scientific validation. Report ranked space groups and, when requested, candidate rankings with their provenance and limitations. DTW ranks pattern agreement; it does not establish structural identity.

The skill orchestrates CLI tools; external engines and licensed assets remain separate dependencies. Describe only stages actually executed and distinguish numerical protocol checks from historical reproduction. Do not claim improved scientific accuracy or autonomous validation merely because an agent ran the commands.
