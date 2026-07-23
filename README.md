# Heads Collapse, Features Stay: Why Replay Needs Big Buffers

Code for the paper *"Heads Collapse, Features Stay: Why Replay Needs Big Buffers"* (Giulia Lanzillotta, Damiano Meier, Thomas Hofmann — ETH Zürich). Paper: https://openreview.net/forum?id=IdW0d0mRnG

We distinguish *deep* (feature-space) forgetting from *shallow* (classifier-level) forgetting in continual learning with Experience Replay, and show that small replay buffers are enough to prevent deep forgetting, while preventing shallow forgetting requires much larger buffers. We explain this asymmetry by extending Neural Collapse theory to the continual learning setting, and connect it to out-of-distribution detection.

This codebase is a trimmed-down fork of the [Mammoth](https://github.com/aimagelab/mammoth) continual learning framework, keeping only what was needed to produce the paper's results.

## What's here

- **Datasets** (`datasets/`): CIFAR-100, Tiny-ImageNet and CUB-200, each with class-/task-incremental (`seq-*`) and domain-incremental (`inc-*`) variants, plus the pretrained-vs-from-scratch CUB-200 ablation (`seq-cub200-scratch`) and the feature-bottleneck ablation (`*-chunks`).
- **Backbones** (`backbone/`): ResNet18 (from scratch), ResNet50 and ViT (pretrained on ImageNet).
- **Methods** (`models/`): `er_balanced` (Experience Replay), `der` (Dark Experience Replay), `fdr` (Function Distance Regularization), `icarl`, and `sgd` (no-replay lower bound / joint upper bound via `--joint`). `er_metrics` is a variant of ER that additionally logs measurements *during* training rather than only at task boundaries.
- **Metrics**: `utils/feature_forgetting.py` (linear-probe deep/shallow forgetting), `utils/loggers_NC.py` (Neural Collapse metrics).

## Setup

```bash
pip install torch torchvision kornia scikit-learn pandas timm tqdm setproctitle
```

Datasets are downloaded automatically into `./data/` on first use (override with `--base_path`).

## Running a single experiment

```bash
python utils/main.py --dataset=seq-cifar100 --training_setting=class-il \
    --model=er_balanced --buffer_size=500 --lr=0.1 --seed=1000 \
    --log_feature_forgetting=1 --log_NC_metrics=1 --permute_classes=1
```

Run `python utils/main.py --model=<model> --help` to see all options for a given method (dataset, model and lr are always required).

### Datasets and CL settings

The paper studies three continual-learning settings — **class-incremental (CIL)**, **task-incremental (TIL)** and **domain-incremental (DIL)** — combined with three datasets. Which setting you get is a combination of *which dataset* you pick and the `--training_setting` flag:

| Dataset family | Backbone | CIL | TIL | DIL |
|---|---|---|---|---|
| CIFAR-100 (32×32) | ResNet18, from scratch | `--dataset=seq-cifar100 --training_setting=class-il` | `--dataset=seq-cifar100 --training_setting=task-il` | `--dataset=inc-cifar100 --training_setting=class-il` |
| CIFAR-100 (224×224) | ViT, pretrained | `--dataset=seq-cifar100-224 --training_setting=class-il` | `--dataset=seq-cifar100-224 --training_setting=task-il` | `--dataset=inc-cifar100-224 --training_setting=class-il` |
| Tiny-ImageNet | ResNet18, from scratch | `--dataset=seq-tinyimg --training_setting=class-il` | `--dataset=seq-tinyimg --training_setting=task-il` | `--dataset=inc-tinyimg --training_setting=class-il` |
| CUB-200 | ResNet50, pretrained | `--dataset=seq-cub200 --training_setting=class-il` | `--dataset=seq-cub200 --training_setting=task-il` | `--dataset=inc-cub200 --training_setting=class-il` |

Notes:
- `--training_setting` only takes `class-il`/`task-il` — for a domain-incremental run, pick an `inc-*` dataset *and* leave `--training_setting=class-il` (its default); the code raises an error if you combine an `inc-*` dataset with `task-il`, since DIL and TIL are mutually exclusive by construction.
- `seq-cub200-scratch` is the same as `seq-cub200` but with a randomly-initialized (non-pretrained) ResNet50, used for the pretrained-vs-from-scratch ablation.
- `seq-cifar100-chunks` / `inc-cifar100-chunks` is the feature-bottleneck ablation (4 tasks of 25 classes, a ResNet18 with a 10-dimensional bottleneck before the head).
- The backbone is fixed per dataset (not a CLI flag) — see `get_backbone()` in the corresponding `datasets/*.py` file.
- `--optimizer` defaults to `sgd`; the paper's ResNet experiments all use `sgd`, while the ViT experiments (`seq-cifar100-224`/`inc-cifar100-224`) use `--optimizer=adamw` with a much lower `--lr` (e.g. `0.0001` vs. `0.03`–`0.1` for SGD runs). See the paper's "Hyper parameters" table for the exact per-dataset/method learning rates and weight decays used.

### Methods

| `--model` | Method | Extra required flags |
|---|---|---|
| `sgd` | Fine-tuning baseline (no replay). Add `--joint=1` to instead train jointly on all data seen so far (upper bound). | — |
| `er_balanced` | Experience Replay, with a class-balanced first batch. | `--buffer_size` |
| `er_metrics` | Same as `er_balanced`, but additionally logs measurements *during* training (not just at task boundaries) — used to produce the paper's within-task figures. | `--buffer_size` |
| `der` | Dark Experience Replay (distills past logits from the buffer). | `--buffer_size`, `--alpha` |
| `fdr` | Function Distance Regularization (matches softmax outputs on buffer samples). | `--buffer_size`, `--alpha` |
| `icarl` | iCaRL (nearest-class-mean classification + distillation). | `--buffer_size` |

### Logging

Runs write their results as append-only text files under `data/results/<setting>/<dataset>/<model>/logs.txt` (one Python-dict-repr line per run; `<setting>` is `task-il`, `class-il` or `domain-il` depending on the run). Three additional flags control the paper-specific measurements:

- `--log_feature_forgetting=1`: fits a linear probe on frozen features after each task to measure *deep* forgetting, alongside the standard (*shallow*) accuracy — see `utils/feature_forgetting.py`. Written with `result_type` `features_cil`/`features_til` in the same results tree.
- `--log_NC_metrics=1`: computes Neural Collapse metrics (NC1–NC3) on the buffer/train/test features after each task — see `utils/loggers_NC.py`.
- `--store_features=1`: dumps raw features/labels at the end of training (see `ContinualModel.store_features`).

## Launching a grid of experiments (SLURM)

The intended workflow for launching many runs is:

1. **`scripts/prepare_grid.py`** builds a text file of CLI argument combinations (one line per run) by taking the cartesian product of a dict of hyperparameter lists. It currently defines the CIFAR-100 grid used for the paper's buffer-size sweep:

   ```bash
   python scripts/prepare_grid.py --job_folder=data/jobs/
   ```

   This writes `data/jobs/list_cifar100.txt` (and `list_all_grid.txt`). Edit the `grid_combinations` list at the top of the script to add further datasets/sweeps.

2. **`scripts/slurm_sbatcher.py`** takes that job list and submits it as a SLURM array job:

   ```bash
   python scripts/slurm_sbatcher.py --file=data/jobs/list_cifar100.txt --per_job=1 --gpus=1
   ```

   Useful flags: `--dry` (write the sbatch script without submitting), `--at_a_time` (throttle concurrent array tasks), `--account`/`-A` and `--partition`/`-p` (SLURM account/partition, omitted from the script if not set), `--bashrc` (shell script to source for cluster modules / environment activation, e.g. `module load ...` + `source /path/to/venv/bin/activate`).

   **Note:** `scripts/slurm_sbatcher.py` is cluster-agnostic — it doesn't assume any particular account, partition or environment setup. Pass `--account`, `--partition` and `--bashrc` as needed for your cluster.

For quick local runs without SLURM, `scripts/local_launcher.py` runs a job list sequentially/in parallel on the local machine instead.

## Acknowledgments

This codebase builds on the [Mammoth](https://github.com/aimagelab/mammoth) continual learning framework (Buzzega et al., *Dark Experience for General Continual Learning*, NeurIPS 2020; Boschini et al., *Class-Incremental Continual Learning into the eXtended DER-verse*, TPAMI 2022), which provided the training loop, replay buffer and logging infrastructure this project extends.
