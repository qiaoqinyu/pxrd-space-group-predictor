# Public/private/external boundary

## Public core

- SERes PXRD encoder and molecular-embedding fusion.
- Lightweight graph teacher used for training-only classification and alignment.
- Array contracts, training, evaluation and graph-free inference.
- Experimental classification preprocessing and CSD-only training perturbations.
- Constrained DTW scoring for already simulated PXRD profiles, with an optional
  energy-then-DTW selector for already deduplicated candidates.

## Private evidence

- Fixed paper splits, row-level predictions, checkpoints and seed aggregates.
- Manuscript, SI, figure, manifest and checksum production.
- Outcome-selected CSP case tables and diagnostics.

## Downstream adapters and external engines

Public tools prepare conformers, call a pinned clean BOMLIP-CSP snapshot, run the two-stage MACE protocol, deduplicate within space groups and simulate candidate PXRD. Optional PackingSimilarity validation requires a licensed CSD Python API. See `downstream-workflow.md`. External engines, trained models, licensed structures and historical case outputs are not bundled. DFT remains external. DTW ranks pattern consistency; packing validation is a separate post-ranking check.

## Pending decisions

Software licensing, authorship and citation metadata, model-weight redistribution, data assets and Zenodo deposition require separate approval.
