# PXRD Space Group Predictor

## Unified CLI

For a new Linux x86-64 installation, follow the tested
[Conda installation steps and version locks](environments/README.md), then run
`pxrd-sg --help` and `pxrd-sg doctor`.
The `predict`, `run`, `rank`, and `doctor` commands provide a single interface to
the existing adapters. `run` requires an explicit candidate budget and GPU;
prediction alone never starts CSP. See [CLI guide](docs/cli.md) and
[environment profiles](environments/README.md). Download the paper's
[trained classifier](docs/model-weights.md) and supply a local ChemBERTa snapshot.
The weights are Release assets, separate from the source and wheel.
The full GPU workflow uses a separate BOMLIP/MACE Conda environment and requires
a compatible NVIDIA driver and CUDA toolkit. The tested installation uses no
container runtime.

This repository provides the core method used to rank molecular-crystal space groups from PXRD and molecular information.

The code exposes the following reader-facing workflows:

- train a PXRD + molecular-embedding classifier with an optional crystal-graph teacher;
- evaluate ordinary and class-balanced TOP-K recall;
- run graph-free inference from PXRD and molecular embeddings;
- preprocess experimental PXRD for classification;
- rank precomputed candidate PXRD profiles with constrained dynamic time warping (DTW), optionally after an energy filter;
- prepare conformers, run pinned BOMLIP-CSP generation and two-stage MACE relaxation, deduplicate structures and simulate PXRD;
- optionally validate crystal packing with a separately licensed CSD Python API. This step is unavailable without that license and local CCDC installation.

The formal model uses a 3,400-point PXRD grid, a squeeze-and-excitation residual PXRD encoder, cached molecular embeddings, and a lightweight two-layer graph encoder during training. The graph pathway supplies classification and representation-alignment supervision. Evaluated inference logits use only PXRD and the molecular embedding.

## Scope

This repository provides the core software, not a complete paper-reproduction package. It intentionally excludes:

- licensed CSD and ICDD/PDF source records and row-level derivatives;
- molecular identifiers, experimental PXRD arrays, crystal coordinates and CIFs;
- trained checkpoints whose redistribution rights have not been cleared;
- manuscript, figure, Supporting Information and internal evidence builders;
- historical case-specific calculation outputs and proprietary computational engines.

Downstream adapters call separately installed scientific engines. See [the downstream workflow](docs/downstream-workflow.md) for dependencies, fixed parameters and commands. DTW is a PXRD-consistency ranking signal and does not establish crystal-packing identity.

## Start here

```bash
git clone https://github.com/qiaoqinyu/pxrd-space-group-predictor.git
cd pxrd-space-group-predictor
```

1. Follow [the Conda installation guide](environments/README.md).
2. Obtain the [model assets](docs/model-weights.md) and run the environment checks.
3. Use [the CLI guide](docs/cli.md) for prediction, full runs or independent ranking.

Training, array contracts and individual Python adapters are documented in
[advanced usage](docs/advanced-usage.md). They are not required for the CLI path.

## Agent skill

The repository includes [PXRD space-group screening](skills/pxrd-space-group-screening/SKILL.md),
a small agent guide for the CLI. Use the installation steps above.

Open this checkout in an agent that supports project skills. The symlink under
`.agents/skills/` points to the canonical skill. In Codex, invoke
`$pxrd-space-group-screening`; with other agents, provide the full path to
`skills/pxrd-space-group-screening/SKILL.md` and ask the agent to follow it.
Keep the skill with the repository because it calls the repository's CLI tools.
No global skill installation is required.

For example, ask your agent:

> Use the pxrd-space-group-screening skill to predict the top three space groups
> from my observed.txt and supplied molecular SMILES, using the public checkpoint.
> Save the commands, input hashes and predictions in a new runs/ directory.

**The paper's full_pretrained checkpoint for seed 240427 is publicly available.**
The [model guide](docs/model-weights.md) provides the download, SHA256, class
mapping and preprocessing/embedding requirements. This is one trained model,
not an ensemble or all five runs used for aggregate paper results. The agent must
verify the downloaded checkpoint before prediction. Candidate DTW ranking can
run without a classifier checkpoint. The synthetic example tests execution only.

The skill also guides conformer preparation, BOMLIP-CSP generation, two-stage
MACE relaxation, deduplication, simulated PXRD and optional CCDC packing
validation. See [the downstream workflow](docs/downstream-workflow.md) for
commands, dependencies and verification limits. DFT remains a separate workflow.
The wheel includes the CLI adapters and installs the skill, guides, configs,
example generator and environment profiles under
`<environment>/share/pxrd-space-group-predictor/`.

## Release status

This repository is publicly maintained at https://github.com/qiaoqinyu/pxrd-space-group-predictor. Transfer to the pic-ai-robotic-chemistry organization is planned only after the author requests it. The designated classifier is available in the `weights-v1` Release. Software/model licensing, citation metadata and data/Zenodo deposits remain pending. Public visibility does not grant an open-source license.
