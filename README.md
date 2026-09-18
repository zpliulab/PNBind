# PNBind

This repository contains the public training, inference, and evaluation package for PNBind, a residue-level predictor of protein–nucleic-acid binding sites.

The release provides the model definitions, training loop, preprocessing-manifest utility, inference code, test data, inference-only model weights, example precomputed features, and the artifacts needed to reproduce the reported benchmark values. Optimizer state, server-local intermediate files, and full training checkpoints are not distributed.

## Reproduce the PNBind rows in Table 2

This is the fastest and most portable reproduction path. It runs on CPU and does not require model weights.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-eval.txt
python scripts/reproduce_table2.py --check
```

Expected four-decimal output:

| Dataset | Chains | Residues | Threshold | MCC | F1 | Recall | Precision | Specificity | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DNA-Test-129 | 129 | 37,515 | 0.58 | 0.5860 | 0.6111 | 0.6237 | 0.5991 | 0.9735 | 0.9525 | 0.6361 |
| DNA-Test-181 | 181 | 75,088 | 0.55 | 0.4252 | 0.4432 | 0.5766 | 0.3600 | 0.9543 | 0.9198 | 0.4171 |
| RNA-Test-117 | 117 | 37,345 | 0.40 | 0.3512 | 0.3696 | 0.6081 | 0.2655 | 0.9033 | 0.8759 | 0.2997 |
| RNA-Test-285 | 285 | 45,317 | 0.34 | 0.4161 | 0.4722 | 0.4894 | 0.4562 | 0.9400 | 0.8679 | 0.4438 |

The script writes both the reproduced values and an explicit residue-alignment log under `results/`.

## Train from precomputed graph features

The training entry point is `scripts/train.py`. It instantiates the published PNBind model family, uses weighted binary cross-entropy, AdamW, ReduceLROnPlateau learning-rate scheduling, gradient clipping, early stopping, deterministic seeding, and best-validation-checkpoint saving. The complete default hyperparameter set is in `configs/train_example.json`.

Training uses per-chain graph caches and ESM3 layer tensors rather than server-local paths. Each graph must contain the fields used by `pnbind.data.load_feature_graph`, including residue labels (`y`); each matching ESM3 layer file must contain `esm3_layers`. Build deterministic train/validation manifests from these public-format precomputed features:

```bash
pip install -r requirements-training.txt
python scripts/prepare_training_manifest.py \
  --graph-dir /path/to/train_graphs \
  --esm3-layers-dir /path/to/train_esm3_layers \
  --train-output manifests/train.jsonl \
  --validation-output manifests/validation.jsonl \
  --seed 5002
```

Run training with:

```bash
python scripts/train.py \
  --config configs/train_example.json \
  --train-manifest manifests/train.jsonl \
  --validation-manifest manifests/validation.jsonl \
  --output-dir runs/dna_seed5002
```

`runs/dna_seed5002/best.pt` contains the best validation checkpoint and `history.json` records the loss and learning-rate trajectory. The repository does not bundle the large, regenerable graph/PLM feature cache; benchmark source data can be obtained from the public benchmark providers, then converted to this documented feature format.

## Evaluation protocol

The packaged probability arrays are mean probabilities from eight released checkpoints per nucleic-acid task. Binary metrics use the fixed thresholds reported with the manuscript. In the original analysis these thresholds were selected independently on each benchmark by scanning from 0.01 to 0.99 in steps of 0.01 and maximizing MCC. The reproduction script never re-optimizes them.

Predictions and labels are matched by chain ID. If a chain-level length differs, both arrays are prefix-aligned to the shorter length. This affects 9 chains in DNA-Test-129, 10 in DNA-Test-181, 3 in RNA-Test-117, and none in RNA-Test-285; every adjustment is written to `results/alignment_log.csv`.

`results/table2_all_methods.csv` preserves the complete comparison table. This package directly reproduces only the PNBind rows; baseline rows are retained as reported comparison artifacts.

## Model inference from precomputed features

Model checkpoints are hosted as GitHub Release assets because they exceed the normal Git file-size limit. After release `v1.0.0` is available:

```bash
pip install -r requirements-inference.txt
python scripts/download_checkpoints.py --tag v1.0.0 --task DNA
python scripts/infer_precomputed.py \
  --task DNA \
  --checkpoint-dir checkpoints \
  --device cuda:0 \
  --check
```

The repository includes one DNA and one RNA precomputed example. Pass `--graph` and `--esm3-layers` to use other features in the same format. Checkpoints are loaded one at a time, so only one model needs to reside on the GPU at once.

The checkpoint files are inference-only exports. They contain the model state and the minimal architecture configuration required to restore it, but no optimizer, epoch, validation record, machine path, or training command.

## Data and artifact layout

```text
benchmarks/data/          four public benchmark test sets
benchmarks/predictions/   per-chain PNBind ensemble probabilities
checkpoints/              manifest and downloaded release assets
examples/                 two precomputed inference examples
pnbind/models/            model definitions needed for inference
results/                  expected and reproduced benchmark tables
scripts/                  evaluation, download, verification, and inference CLIs
configs/                  published training hyperparameters
pnbind/training.py        training loop, loss, optimizer, scheduler, and checkpoint logic
```

The four test files follow the three-line FASTA/label format distributed by the MegSite repository. The repository copy, filenames, and checksums are fixed in this release so that future upstream changes do not silently change the benchmark.

## Verification

Before publishing a modified copy, run:

```bash
python scripts/verify_release.py
```

This checks required files, repository-size limits, accidental local paths or credential-like strings, and the Table 2 reproduction.

## Availability boundary

This release supports exact reproduction of the reported PNBind benchmark rows, checkpoint-level inference, and retraining from the documented precomputed graph and ESM feature representation. It does not distribute server-local intermediate files, optimizer state, or full training checkpoints.
