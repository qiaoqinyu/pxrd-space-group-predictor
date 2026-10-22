# PXRD Space Group Predictor

Predict molecular-crystal space groups from an experimental powder X-ray diffraction (PXRD) pattern and a molecular SMILES string. Use the predicted shortlist to guide crystal structure prediction, then rank candidate structures by energy and agreement with the experimental PXRD pattern.

The project provides a command-line interface (`pxrd-sg`) and a companion skill for AI agents.

```text
Experimental PXRD + molecular SMILES
                 ↓
       Space-group prediction
                 ↓
   Candidate generation with BOMLIP-CSP
                 ↓
        MACE structure relaxation
                 ↓
  Deduplication and PXRD simulation
                 ↓
   Energy preselection and DTW ranking
                 ↓
       Ranked candidates and CIFs
```

DTW (dynamic time warping) measures agreement between diffraction profiles. Candidate rankings support further structure assessment; they do not establish structural identity.

## What you can run

| Command | Inputs | Outputs |
| --- | --- | --- |
| `pxrd-sg predict` | Raw PXRD, molecular SMILES and model assets | Space-group probabilities and a ranked shortlist |
| `pxrd-sg run` | Prediction inputs, engine assets and an explicit search budget | Candidate structures, energies, DTW rankings and selected CIFs |
| `pxrd-sg rank` | Raw PXRD and a relaxed-candidate manifest | Candidate rankings and selected CIFs, without classifier inference |

`pxrd-sg doctor` checks dependencies and supplied assets before computation. Runs record input/model hashes, settings and output checksums. Use `--resume` to reuse verified completed stages.

## Quick start

```bash
git clone https://github.com/qiaoqinyu/pxrd-space-group-predictor.git
cd pxrd-space-group-predictor
```

1. Follow the [Conda installation guide](environments/README.md). Prediction and ranking use one environment; candidate generation and MACE relaxation use a separate GPU engine.
2. Download and verify the [required model assets](docs/model-weights.md).
3. Follow the [CLI examples](docs/cli.md) to check the environment and run prediction, a full workflow or independent ranking.

Supply PXRD as two columns: angle in degrees 2theta and linear intensity. The current CLI requires a wavelength of 1.54184 Å. Full runs also require a candidate count and GPU selection.

The tested platform is Linux x86-64 with Python 3.10. MACE relaxation requires a compatible NVIDIA GPU, driver and CUDA toolkit. macOS and native Windows installations remain unverified. No container is required.

Start with the [one-candidate synthetic example](docs/cli.md#synthetic-execution-example) to check installation. Its artificial pattern tests execution, not predictive accuracy or experimental structure recovery.

## Use with an AI agent

The [PXRD screening skill](skills/pxrd-space-group-screening/SKILL.md) guides an agent through installation, input checks, CLI commands and result inspection. It uses the same CLI as a manual run.

Open the checkout in Codex and invoke `$pxrd-space-group-screening`. For other agents, provide the path to `skills/pxrd-space-group-screening/SKILL.md`. No global skill installation is required.

Example request:

> Follow the PXRD screening skill to predict the top three space groups from my experimental PXRD file and molecular SMILES. Use the public checkpoint, verify the inputs and model assets, and save predictions and logs in a new output directory. Run prediction only.

Candidate generation and GPU relaxation require a separately specified compute budget.

## Models, data and licensing

The [weights-v1 release](https://github.com/qiaoqinyu/pxrd-space-group-predictor/releases/tag/weights-v1) provides one classifier used in the paper. It is a single trained model, not an ensemble or the complete set of training runs. ChemBERTa and MACE are obtained separately from their upstream sources. See the [model guide](docs/model-weights.md) for downloads, verification and applicable terms.

The repository does not distribute licensed CSD/PDF database records or restricted study datasets and historical calculation outputs. Users can supply their own inputs and export generated candidate CIFs. The public prediction and ranking workflow does not require a CSD license. Optional reference-based packing validation requires a separately licensed CCDC environment.

Formal software and classifier license terms, citation metadata and archival deposition remain pending. Public visibility does not grant an open-source license. Upstream model and engine licenses apply separately.

## Documentation and validation

- [Installation and tested environments](environments/README.md)
- [CLI commands, examples and resume behavior](docs/cli.md)
- [Model downloads and provenance](docs/model-weights.md)
- [Downstream methods and optional packing validation](docs/downstream-workflow.md)
- [Training, array formats and direct adapters](docs/advanced-usage.md)
- [Paper-to-code correspondence](docs/paper-code-map.md)

Validation includes the 58-test suite and a one-candidate GPU workflow, with stage resume, independent ranking and output-hash checks. These checks establish software execution in the tested environments, not reproduction of the paper's full results. See the environment guide for the validation scope.
