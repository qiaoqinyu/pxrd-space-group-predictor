# Downstream crystal workflow

These adapters connect the existing prediction and DTW tools to BOMLIP-CSP,
MACE relaxation, within-space-group deduplication and optional licensed packing
validation. They preserve the reported numerical settings. They do not supply
trained weights, licensed structures or the historical calculation environment.
Protocol agreement does not imply bitwise reproduction of historical results.

## Environment and inputs

Use an environment containing the project's dependencies plus RDKit, pymatgen
and ASE (`.[structures]`). Generation additionally needs BOMLIP-CSP's dependencies;
relaxation needs its batchopt/MACE stack, CUDA, cuEquivariance and Ninja. Inspect
the upstream environment requirements before installing anything. Invoke Python
from that environment; the relaxation adapter prepends its bin directory to PATH
and its isolated upstream batchopt source to PYTHONPATH. Do not combine arbitrary
PyTorch/CUDA/cuEquivariance versions.

Obtain a **clean** clone of
https://github.com/pic-ai-robotic-chemistry/BOMLIP-CSP at commit
`2caf530c8aa2ca14230264bfe69bfbcd6952b41e`. Both generation and relaxation make
tracked-file snapshots in new output directories. They reject a dirty or different
checkout. Original checkouts remain unchanged. MACE's model path is changed only
inside the snapshot; three hardcoded model calls are replaced with an environment
lookup, with before/after hashes recorded. Upstream MIT licensing remains upstream.

Supply a trusted local MACE-OFF23-small model with SHA256
`165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f`.
The adapter checks this hash before loading. There is no automatic model download.
Generation and MACE runs require explicit compute authorization from the operator;
a prediction request alone must not launch them. Choose available hardware first.

Use single-component molecular SMILES and explicit concrete space-group numbers.
Confirm the shortlist with the user before a costly search. `Others` is an
aggregate prediction class and cannot be passed as a space group. The retrospective
paper cases were selected with known shortlist coverage; new predictions do not
guarantee that coverage.

## Generate candidates

```bash
python tools/generate_crystal_candidates.py --smiles CCO --sample-id ethanol --space-groups 1 2 --bomlip-root /path/to/BOMLIP-CSP --output-dir runs/task/generated --count 5000 --workers 8
```

ETKDGv3 requests 20 explicit-hydrogen conformers, with seed 240430 and RMS pruning
0.1 Å. UFF optimizes them all. Converged conformers come first, followed by lower
UFF energy and lower conformer identifier. The first conformer is passed to
BOMLIP-CSP without further conformer generation, with Z′ = 1. Generation starts
at seed 1 in each group. Each group requests 5,000 successful structures; a smaller
`--count` is an execution smoke test, not the paper candidate budget. The timeout
bounds each group's retries. Failed or partial runs are not complete manifests.

`generation.json` records atom count, commands, settings and final manifest hash.
`candidates.csv` contains exactly one sample, unique string IDs across groups,
requested space groups, CIF paths and SHA256 hashes, blank energies and generated
status. Space-group fields describe the generation pool, not independently
redetermined final symmetry. Keep output directories intact when reusing manifests.

## Two-stage MACE relaxation

```bash
python tools/relax_crystal_candidates.py --manifest runs/task/generated/candidates.csv --bomlip-root /path/to/BOMLIP-CSP --model /path/to/MACE-OFF23_small.model --output-dir runs/task/relaxed --gpu 0 --workers 8
```

The atom count per molecule comes from the hash-linked `generation.json` beside
the manifest. `--generation-metadata` selects a relocated file. An optional
`--atoms-per-molecule` is an assertion and must agree; a divisor of the cell's
atom count alone is not sufficient to define molecular energy normalization.
The manifest sample, metadata hash link and cell divisibility are checked.

Stage 1 uses UnitCellFilter/BFGSFusedLS at 0.0006 eV Å⁻³; stage 2 uses
UnitCellFilter/BFGS at zero pressure. Each stage uses fmax 0.01 eV Å⁻¹ and at
most 3,000 steps. Batch size is zero, with two CPU threads per worker and
cuEquivariance enabled. Defaults use eight workers on one explicitly selected GPU;
change workers to fit hardware, recording the change. Upstream optimizer equations
and stopping rules remain unchanged.

`relaxed.csv` preserves candidate identities and records **kJ/mol per molecule**,
not eV per cell. Upstream JSON energies already have this conversion. A stage at
or near the step limit produces `step_limit`; other outputs are `completed`.
These flags do not independently certify force convergence. Inspect stage logs
and JSON outputs before scientific interpretation. Preserve flagged candidates
as in the reported screening protocol; report their count instead of silently
claiming all structures converged. A missing/invalid output fails the command.

## Deduplicate and simulate

```bash
python tools/process_crystal_candidates.py --manifest runs/task/relaxed/relaxed.csv --output-dir runs/task/processed
```

Within each requested space group, sort by energy then string ID. Use
StructureMatcher with ltol 0.2, stol 0.3, angle tolerance 5°, ElementComparator,
ignored H, primitive_cell=False, scale=True and attempt_supercell=False. Assign
each structure to the first directly matching retained representative. There is
no transitive closure, cross-group merging, energy/density prefilter or early stop.
The representative manifest and mapping preserve that provenance.

Simulate Cu Kα PXRD at 1.54184 Å on 3,400 points, 2.00–69.98° with step 0.02°;
query peaks over 2–70°. Apply pseudo-Voigt broadening with FWHM 0.15° and Lorentz
fraction 0.5, then linear max100 scaling to float32. Export aligned
`candidates_raw.npy`, `energies.npy` and `ids.npy` for the existing DTW command.

## Energy and DTW selection

```bash
python tools/score_pxrd_candidates.py --reference observed_raw.npy --candidates runs/task/processed/candidates_raw.npy --energies runs/task/processed/energies.npy --candidate-ids runs/task/processed/ids.npy --start-index 0 --stop-index 3400 --output runs/task/scores.csv
```

The example assumes full measured support. Interpolate the raw experimental
profile onto the same grid without the classification background filter and
choose actual measured crop indices. First crop both profiles; then subtract
the experimental minimum within that interval. Both profiles undergo max100,
five-point maximum pooling and independent L2 normalization. Discard an incomplete
final bin. Use linear intensities, band radius 30, absolute local differences,
zero penalty/endpoint relaxation, and divide accumulated cost by sqrt(pooled
length). Keep up to 200 lowest-energy representatives, then select up to ten by
DTW. Energy ties use ID; DTW ties use energy then ID. Reference packing labels
never enter this selection. Fewer available candidates remain a smaller set.

## Optional packing validation

```bash
python tools/validate_crystal_packing.py --manifest runs/task/processed/representatives.csv --reference unoptimized_experimental.cif --output-dir runs/task/packing --timeout 30
```

Run this separately in a compatible, licensed CSD Python API environment. Only
standard-library dependencies are needed by this adapter; the reference and
candidate paths must be readable by that environment. PackingSimilarity requires
all 15 molecules in the shell, distance tolerance 0.2, angle tolerance 20°,
no molecular differences, artificial inversion enabled, and ignored hydrogen
positions/counts and bond counts. Each candidate runs in a bounded subprocess.
Errors and timeouts have null hit values and produce a nonzero command exit;
they are unresolved, not negative matches. The result is post-ranking validation
and cannot change the DTW shortlist. Supply a selected-subset manifest if only
the final ten should be validated. DFT remains a separate workflow.

## Verification boundary

Tests cover manifests, hash/unit checks, molecular normalization, synthetic
structures, subprocess failure/timeout and the existing classifier/DTW workflows.
Execution smoke tests do not establish predictive accuracy or reproduce the
paper's full CSP results. Trained prediction weights and licensed datasets remain
separate assets; do not substitute synthetic/random weights for missing models.

The verification run included one real ethanol candidate through generation,
two-stage MACE, deduplication, simulation and DTW. Original-helper comparisons
covered three synthetic structures. A licensed Windows CSD API self-match returned
15 matched molecules and zero RMSD, but its Python process stalled during exit.
Clean termination of that platform's complete packing CLI remains unverified.
The timeout path is regression-tested and records an unresolved result.
