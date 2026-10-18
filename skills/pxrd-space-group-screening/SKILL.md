---
name: pxrd-space-group-screening
description: Rank molecular-crystal space groups from experimental PXRD and SMILES, or rank existing candidate PXRD profiles, using this repository's trained model and DTW tools.
---

# PXRD-guided space-group screening

Use the existing CLI tools to execute the requested prediction or candidate-ranking task. Resolve any symlink to this skill directory first, then resolve the repository root from that directory (`../..`). Run commands from that root in its configured Python environment; inspect `README.md`, `pyproject.toml` and each tool's `--help` when needed.

## Establish inputs and model provenance

For prediction, obtain experimental two-column 2theta/intensity text, canonical isomeric SMILES, stable sample IDs and a compatible trained checkpoint. Confirm that angles are degrees 2theta and record the radiation/wavelength used to obtain each pattern. The paper protocol uses Cu Kα radiation; different experimental conditions require a compatibility assessment, not silent conversion. Read `configs/paper_protocol.json` and the README's array contract. Synthetic examples validate execution only; they are not trained paper models.

Trained weights are not distributed in this repository; obtain a compatible checkpoint separately. There is currently no public weight download. Record checkpoint SHA256, model configuration, class mapping and training provenance. If unavailable, report the missing asset and stop prediction; candidate ranking can still run independently. Never replace a missing trained checkpoint with random weights. For a trusted historical SE-Res/simple-graph checkpoint, `tools/import_legacy_checkpoint.py` creates a new schema-1 copy with restricted loading and exact graph-key renaming. Supply a verified hash, configuration, class mapping and provenance label. Confirm prediction parity against its original implementation before scientific use. A single checkpoint does not reproduce aggregate results from multiple training runs.

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

## Deliver a traceable result

Save commands, code revision, input/model hashes, preprocessing and embedding settings, class mapping, measured support, output paths and checks in the task output directory. Respect the redistribution terms for supplied data and checkpoints; share only assets authorized for release. Distinguish software execution checks from scientific validation. Report ranked space groups and, when requested, candidate rankings with their provenance and limitations. DTW ranks pattern agreement; it does not establish structural identity.

The supported workflow ends at prediction or ranking supplied candidates. CSP generation, MACE relaxation, PackingSimilarity and DFT remain external workflows. Report that boundary when a request needs those stages; use their established procedures and execution authorization rather than claiming this skill runs them.
