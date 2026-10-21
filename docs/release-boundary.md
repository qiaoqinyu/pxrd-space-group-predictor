# Public/private/external boundary

## Public core

- SERes PXRD encoder and molecular-embedding fusion.
- Lightweight graph teacher used for training-only classification and alignment.
- Array contracts, training, evaluation and graph-free inference.
- Experimental classification preprocessing and CSD-only training perturbations.
- Constrained DTW scoring for already simulated PXRD profiles, with an optional
  energy-then-DTW selector for already deduplicated candidates.

## Private evidence

- Fixed paper splits, row-level predictions, unpublished checkpoints and seed aggregates.
- Manuscript, SI, figure, manifest and checksum production.
- Outcome-selected CSP case tables and diagnostics.

## Public model asset

The `weights-v1` Release provides the full_pretrained classifier for seed 240427,
its configuration, provenance and SHA256 checksums. See `model-weights.md`.
This designated checkpoint is authorized for public release. It contains learned
tensors and model metadata, with no optimizer state or row-level dataset records.
The file is separate from Git source history and Python packages. Other models
and licensed datasets are not included. Formal software/model licensing remains
pending; publication does not assign a new license.

## Downstream adapters and external engines

Public tools prepare conformers, call a pinned clean BOMLIP-CSP snapshot, run the two-stage MACE protocol, deduplicate within space groups and simulate candidate PXRD. Optional PackingSimilarity validation is available only to users with a separate CSD license and a compatible CCDC Python API. The public repository does not provide CSD records, the CCDC API or a license. See `downstream-workflow.md`. External engines, trained models, licensed structures and historical case outputs are not bundled. DFT remains external. DTW ranks pattern consistency; packing validation is a separate post-ranking check.

## Pending decisions

Software/model licensing, authorship and citation metadata, additional model releases, data assets and Zenodo deposition require separate approval.
