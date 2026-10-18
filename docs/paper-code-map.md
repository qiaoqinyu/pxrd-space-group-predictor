# Correspondence to the current manuscript and Supporting Information

Baseline: 18 October 2026, manuscript
`乔钦禹文章-2026.10.18_agent_workflow_reviewed.docx` and
`乔钦禹文章Supporting_Information_2026.10.18_agent_workflow_reviewed.docx`. This document supersedes the protocol descriptions in
the September staging audit, which remains a historical record.

| Paper component | Public implementation | Verification or boundary |
| --- | --- | --- |
| SI-3 experimental classifier preprocessing | `src/data/pxrd.py`, `tools/prepare_pxrd.py` | Interpolation, zero fill, fifth-percentile/601-point baseline, nearest padding, linear max100. |
| SI-3 CSD training augmentation | `augment_csd_profile`, `ArrayDataset` | Independent probability 0.5 for noise and background; no experimental or validation augmentation. |
| SI-3 molecular representation | `tools/embed_smiles.py` | Pinned ChemBERTa revision, 192-token attention-mask mean, float32. Canonical SMILES preparation is external. |
| SI-4 model and graph alignment | `src/models`, `src/training/losses.py` | SERes/simple topology unchanged; inference does not require graphs. |
| SI-5 training and controls | `tools/train.py --stage` | Stage-specific defaults, explicit checkpoint initialization, optimizer reset, no test loader. |
| SI-6 ordinary/balanced recall and macro-F1 | `src/evaluation/metrics.py`, `tools/evaluate.py` | Stable descending logits; summaries plus per-class, confusion and rank-coverage JSON. |
| SI-6 five-seed paired criterion | `paired_seed_summary` | Same five seed IDs, unrounded recall fractions, sample standard deviation; mean gain >=0.02 and >=4 positive pairs. |
| SI-8 DTW | `src/runtime/pxrd_dtw.py` | Linear intensities; Euclidean local cost; radius 30/window 31; normalize by sqrt(length). |
| SI-8 energy-to-DTW selection | `screen_candidates`, `tools/score_pxrd_candidates.py` | Already deduplicated input; select 200 by energy/ID, then ten by DTW/energy/ID. |
| SI-2 fixed split and molecular exclusion | External data preparation | No random split substitutes for the fixed 512/266/1655 paper protocol. |
| SI-7 generation and MACE relaxation | Downstream CLI adapters | `tools/generate_crystal_candidates.py` and `tools/relax_crystal_candidates.py` call the pinned external engine; models are not bundled. |
| SI-8 packing deduplication/validation and retrospective reference | `crystal_processing.py`, `validate_crystal_packing.py` | DTW agreement is not a packing match; ranking does not inspect reference labels. |
| SI-9 VASP comparison | External DFT calculation | The three-structure result is evidence, not a classifier/DTW implementation. |

## Architecture and class mapping

The two 128-dimensional prediction features concatenate to 256 dimensions.
The formal SERes fusion layer maps this **256-dimensional input to 128 outputs**;
the projections then have 64 dimensions. The historical `fusion_hidden_dim=256`
configuration is used by a different optional CNN branch. It is not an extra
256-wide hidden layer in the formal SERes branch. Changing that layer would
change the verified network and checkpoint shapes. The generic API preserves
the original 4,833,230-parameter topology.

The paper class order is SG1, SG2, SG4, SG7, SG14, SG18, SG19, SG29, SG33,
SG61, Others. The configuration stores Others as `null`, whereas the internal
historical label was -1. Dataset labels remain zero-based indices (Others=10).
Others aggregates several groups; it is not a usable space-group number for CSP.

Graphs are supplied as 92-dimensional atom features with up to 12 periodic
neighbors within 8 Å and 41 Gaussian distance features (0–8 Å, 0.2 Å spacing).
The NPZ interface requires already constructed features and padded graphs.
It does not load CIFs, generate those features or recover licensed source rows.

## Training stages

| Setting | `--stage csd-pretrain` | `--stage experimental` (default) |
| --- | --- | --- |
| Maximum epochs | 150 | 120 |
| Train/evaluation batch size | 128 / 256 | 32 / 128 |
| Encoder/head learning rate | 1e-4 / 1e-4 | 3e-6 / 1e-4 |
| Weight decay | 1e-4 | 1e-4 |
| Scheduler monitor | ordinary validation TOP-3 | ordinary validation TOP-3 |
| Checkpoint selection | ordinary validation TOP-1 | ordinary validation TOP-3, then lower loss |
| PXRD augmentation | enabled on training data only | disabled |

Use `--config configs/model.example.json` to fix the paper dimensions and all
eleven classes. Pretraining uses seed 240427. Experimental comparisons use
240427, 240428, 240429, 240430 and 240431. Each adapted run loads the same
pretraining checkpoint using `--init-checkpoint`; the optimizer and scheduler
start afresh. The Experimental-only control omits this argument. The SMILES-only
control uses `--prediction-modality smiles` in **both stages** and a matching
model configuration, retains graph supervision and does not instantiate PXRD.
The no-graph control uses `--no-graph-supervision` to zero graph CE and alignment.
The optional PXRD-only API is a generic facility, not a reported experiment.

AdamW head parameters include `multi_fuse`, `multi_head`, `multi_proj`,
`graph_head` and `graph_proj`. The scheduler uses factor 0.3, patience 8 and
minimum LR 5e-7. With patience 8, the learning rate drops after nine
consecutive validation evaluations without improvement; early stopping uses patience 25. These are exposed CLI values,
not an assertion that new runs reproduce the historical checkpoints.

CSD perturbations use Gaussian sigma 0.002 times the current maximum and
constant/exponential background amplitudes bounded by 0.005/0.010 times that
maximum. The exponential term is exp(-3 position). Each operation independently
occurs with probability 0.5 and restores the original maximum and nonzero-span
padding. Optional integer `row_index` entries preserve original row identities
for deterministic augmentation; otherwise local dataset indices are used.

Balanced recall and macro-F1 average classes with positive test support, as in
the original implementation. All eleven paper test classes have positive
support. Custom partial-class tests report `active_classes` and null recalls
for absent classes, rather than implying an eleven-class evaluation.
`classification_details.json` includes ordinary rank coverage for K=1..10.
The paired-seed helper expects already matched evaluation scopes; it cannot
establish molecule-disjoint membership from metric values.

## Candidate scoring and external stages

Candidate and experimental profiles must already share the grid 2.00–69.98°,
step 0.02°, and measured angular support. Crop both with `--start-index` and
`--stop-index` before subtracting the experimental minimum or normalizing. Simulation uses Cu Kα 1.54184 Å, pseudo-Voigt
FWHM 0.15° and Lorentz fraction 0.5. This package simulates profiles with `tools/process_crystal_candidates.py` before scoring.

After cropping, the minimum experimental intensity within the measured range
is subtracted from the experimental profile only. This is separate from the
classifier's rolling-percentile background subtraction. Both profiles use linear max100
intensities, five-point max pooling and L2 normalization. The incomplete final
bin is discarded. Scalar Euclidean cost is absolute difference; the result is
the **sum of costs along the optimal path divided by sqrt(pooled length)**,
not the square root of the sum of squared costs. Penalty and psi are zero.

When energy metadata are supplied, `--energy-limit 200 --top-count 10` applies
the paper sequence. The output has one row per retained candidate and a
`selected` flag for the final ten. Fewer than 200 available inputs give a smaller
set; no structures are fabricated or added. Identifiers are unique strings and
sort lexicographically for ties. The full scorer without metadata is separate
from this selection mode.

Downstream adapters implement StructureMatcher compression and simulation and call pinned BOMLIP/MACE and licensed PackingSimilarity. VASP remains external. `docs/downstream-workflow.md` gives launch instructions; `configs/paper_protocol.json` records numerical settings. In particular, independent matches require all 15
molecules in the packing shell, and retrospective reference selection uses
only matches within the final ten. The historical nine-case SI figures concern
full candidate pools and are not the current 200-candidate screening panels.

## Input validation

Classification preprocessing rejects profiles with no positive signal after
background correction. Array loading rejects rows with no positive signal in
float32 and reports their zero-based row indices. CSD augmentation rejects
profiles with no positive signal before drawing noise or background values.
These checks reject unusable inputs without changing valid-profile processing.

## Verification scope

Regression tests exercise numerical protocols, input contracts, checkpoint
round trips, one-epoch synthetic training, fine-tuning and graph-free prediction.
The optional reference test compares DTW against dtaidistance Python/C routines.
These tests do not retrain the paper model, validate its scientific accuracy,
recompute the CSP/DFT results or redistribute private data/checkpoints.
