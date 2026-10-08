# Changelog

## 0.1.0.dev1 — manuscript/SI synchronization (2026-10-08)

- Aligned DTW with linear intensities, Euclidean local cost and window 31 semantics.
- Added experimental classification preprocessing and CSD-only augmentation.
- Added explicit pretraining/adaptation stage defaults and corrected the graph
  projection learning-rate group.
- Added Others mapping, stable logit ranking, macro-F1, per-class/confusion and
  rank-coverage outputs, and a five-seed paired comparison helper.
- Added deterministic 200-to-ten screening for pre-deduplicated inputs.
- Documented correspondence to the current manuscript and full SI, preserving
  the existing public-core/external-CSP boundary and network topology.

## 0.1.0.dev0

- Rejected non-finite features, logits and losses, invalid graph indices and masks, and degenerate DTW profiles; training now rejects non-finite gradients.
- Preserved explicit or inherited class counts during subset fine-tuning and added optional checkpointed class-to-space-group mappings to prediction outputs.
- Switched checkpoint loading to the restricted weights-only loader while retaining schema-1 compatibility.
- Added input-contract regressions and a synthetic training, fine-tuning, evaluation, inference and DTW workflow test.
- Created a clean staging package for the reader-facing XRayPro space-group screening code.
- Added the formal SERes PXRD encoder, molecular-embedding fusion and training-only graph teacher.
- Added graph-free inference, labeled evaluation and constrained PXRD-DTW candidate scoring.
- Added a pinned ChemBERTa embedding entry point and synthetic smoke workflow.
- Excluded paper evidence production, restricted data, model weights and case-specific CSP implementation.

This entry records staging work only. It is not a public software release.
