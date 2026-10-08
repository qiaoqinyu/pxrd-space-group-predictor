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

## External CSP

Candidate generation, MACE relaxation, packing-basin deduplication and independent packing validation are outside this repository. The public DTW utility consumes candidate profiles; it does not generate or validate crystal structures.

## Pending decisions

Software licensing, authorship and citation metadata, repository visibility, model-weight redistribution, data assets and Zenodo deposition require separate approval.
