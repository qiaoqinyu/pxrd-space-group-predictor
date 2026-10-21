# Manuscript synchronization — 8 October 2026

> Historical record: this describes checks at the date in the title/body, not
> current installation status. Use [the environment guide](../environments/README.md)
> and [CLI guide](cli.md) for current instructions.


This update aligns the public core with the current manuscript and complete SI.
See [paper-code-map.md](paper-code-map.md) for methods, class mapping and external-stage boundaries.
The September staging audit remains a historical record.

## Changes

- Correct DTW to linear intensity preprocessing and absolute local cost, with radius 30 and accumulated-cost normalization by sqrt(length).
- Add energy-200 then DTW-10 screening with deterministic ties and unique candidate identifiers.
- Add experimental classification preprocessing and deterministic CSD-only augmentation.
- Separate CSD pretraining and experimental defaults; include graph_proj in the head learning-rate group.
- Preserve the original network topology and support the paper's eleven classes, including Others.
- Export per-class recall, macro-F1, confusion matrices, rank coverage and a five-seed paired comparison helper.

## Verification

On lab-wsl2, Python 3.10 / torch 2.4.1 passed 31 tests and skipped one optional dtaidistance test.
The matsci environment with torch 2.8.0 / dtaidistance 2.4.0 passed all 32 tests.
DTW agrees with dtaidistance's Python and C implementations.
Read-only differential checks against the original research source passed 40 exact augmentation cases,
12 experimental preprocessing cases (allowing float32 log-cache round-trip error), and four DTW preprocessing cases.
The CLI help commands, compilation and minimal graph-free example passed.

These are implementation checks with synthetic inputs. They do not establish a new scientific reproduction,
retrain the model, download embeddings, or recompute CSP/DFT results. Those stages and licensed data remain external.
