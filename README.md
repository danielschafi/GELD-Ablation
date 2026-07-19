# GELD Ablation Study

Ablation experiments for [GELD](https://github.com/CIAM-Group/NCO_code) (Global-view Encoder and Local-view Decoder), a neural TSP solver. This repo isolates the contribution of **decoding** (Stage 1) and **refinement** (Stage 2) by swapping each component independently.

Based on the upstream GELD implementation; see the original paper for model architecture and training strategy.

## What is ablated?

GELD inference has two stages:

| Stage | Parameter | Options | Description |
|-------|-----------|---------|-------------|
| **Stage 1** (decoding) | `stage_1` | `Neural` | GELD greedy + neural beam search (default GELD decoder) |
| | | `BeamSearch` | Pure distance-driven beam search (no neural decoder) |
| | | `knn-BeamSearch` | Beam search restricted to *k* nearest neighbours (`knn_k=99`) |
| **Stage 2** (refinement) | `stage_2` | `None` | No post-processing |
| | | `Neural-RC` | Path re-combination (PRC) with GELD re-combination |
| | | `BeamSearch-RC` | PRC with beam-search re-combination |

All nine combinations (`3 × 3`) are evaluated. Stage 2 runs PRC with `num_PRC=1000` when enabled.

### Paper comparisons (Section 2.1)

| Experiment | Compare A | Compare B |
|------------|-----------|-----------|
| 2.1.1 | `BeamSearch` + `None` | `Neural` + `None` |
| 2.1.2 | `BeamSearch` + `Neural-RC` | `Neural` + `Neural-RC` |
| 2.1.3 | `Neural` + `Neural-RC` | `Neural` + `BeamSearch-RC` |
| 2.1.4 | `BeamSearch` + `None` | `knn-BeamSearch` + `None` |

### Evaluation protocol

`test.py` runs on **synthetic** instances at sizes `100, 500, 1000, 5000, 10000` and four point distributions: `uniform`, `clustered`, `explosion`, `implosion`. Metrics logged per block:

- **Optimality gap (%)** — student tour length vs. LKH3 optimal
- **Solving time (min)** — wall-clock for the full episode batch

---

## Setup

### Requirements

- Python ≥ 3.9.19
- CUDA-capable GPU (ablation on `N=10000` is memory-heavy; SLURM jobs request 1× A100, 150 GB RAM)
- [uv](https://docs.astral.sh/uv/) package manager

### Install dependencies

From the repo root:

```bash
uv sync
```

This installs PyTorch 2.2.2, NumPy, Matplotlib, pandas, and the other pinned deps from `pyproject.toml`.

### Data

**Training data** (Stage 1 SL pre-training):

1. Download `train_TSP100_n100w-001.txt` from [LEHD](https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD).
2. Place it in `SL_training_data/`.

**Test data** (synthetic ablation benchmark):

Synthetic instances with LKH3 labels follow [INViT](https://github.com/Kasumigaoka-Utaha/INViT). Put them under `Test_data/` (see upstream GELD repo for the expected layout). The tester generates/loads problems at runtime when `env_params["data_path"]` is `None`.

### Pre-trained checkpoint

Ablation evaluation expects a trained GELD checkpoint at:

```
result/pre_trained_model/checkpoint-49.pt   # epoch 49
```

Configure paths in `test.py`:

```python
model_load_path = "result/pre_trained_model"
model_load_epoch = 49
```

Either train the model yourself (below) or copy a released checkpoint into that folder.

---

## Training

GELD uses a **two-stage** training pipeline.

### Stage 1 — SL pre-training (`train.py`)

Supervised learning on TSP-100 sub-path data:

```bash
uv run -m train
```

Key settings: 50 epochs, 1M episodes/epoch, batch size 1024, LR `1e-4`. Checkpoints are saved under `result/<timestamp>_train/`.

### Stage 2 — RL fine-tuning (`train_new.py`)

Beam-search RL fine-tuning on progressively larger instances, starting from the Stage 1 checkpoint.

1. Set the Stage 1 path in `train_new.py`:

   ```python
   model_load_path = "result/<your_stage1_run>"
   model_load_epoch = 49   # or the epoch you want
   ```

2. Run fine-tuning:

   ```bash
   uv run -m train_new
   ```

   Or submit via SLURM:

   ```bash
   sbatch slurm_train.bash
   ```

After training, point `test.py` at the final checkpoint directory and epoch.

---

## Running the ablation

Ablation settings are passed through environment variables read by `test.py`:

- `STAGE_1` — one of `BeamSearch`, `knn-BeamSearch`, `Neural` (default: `Neural`)
- `STAGE_2` — one of `None`, `Neural-RC`, `BeamSearch-RC` (default: `BeamSearch-RC`)

### Single configuration (local)

```bash
STAGE_1=Neural STAGE_2=Neural-RC uv run -m test
```

Each run creates a timestamped folder under `result/`, e.g. `result/20260521_094801_test_Neural_Neural-RC/log.txt`.

### Full grid (9 jobs via SLURM)

`run_ablation.sh` submits a job array covering all combinations, throttled to 2 concurrent tasks:

```bash
./run_ablation.sh
# equivalent to: sbatch slurm_ablation.bash
```

The array maps `SLURM_ARRAY_TASK_ID` → `(STAGE_1, STAGE_2)`:

| Task ID | STAGE_1 | STAGE_2 |
|---------|---------|---------|
| 0 | BeamSearch | None |
| 1 | BeamSearch | Neural-RC |
| 2 | BeamSearch | BeamSearch-RC |
| 3 | knn-BeamSearch | None |
| 4 | knn-BeamSearch | Neural-RC |
| 5 | knn-BeamSearch | BeamSearch-RC |
| 6 | Neural | None |
| 7 | Neural | Neural-RC |
| 8 | Neural | BeamSearch-RC |

Logs go to `slurm_logs/ablation_<jobid>_<task>.log`.

Adjust partition, GPU, memory, and mail settings in `slurm_ablation.bash` for your cluster.

---

## Getting ablation results

After all nine runs finish, aggregate logs with three scripts (run from repo root):

### 1. Parse raw logs → long-form CSV

```bash
uv run python parse_logs.py
```

**Output:** `result/summary.csv`

Columns: `config`, `distribution`, `size`, `gap_pct`, `time_min`  
One row per (configuration × distribution × size) block.

### 2. Build paper tables (mean over distributions)

```bash
uv run python make_summary_table.py
```

**Outputs:**

| File | Content |
|------|---------|
| `result/summary_mean.csv` | Long-form means per config and size |
| `result/summary_gap_table.csv` | Pivot: optimality gap (%) |
| `result/summary_time_table.csv` | Pivot: solving time (min) |
| `result/summary_gap_table.tex` | LaTeX tabular for gap |
| `result/summary_time_table.tex` | LaTeX tabular for time |

Tables average over the four distributions and list all nine configs × five problem sizes.

### 3. Plot gap and time vs. problem size

```bash
uv run python plot_logs.py              # single figure: result/comparison.png
uv run python plot_logs.py --separate   # result/comparison_gap.png + comparison_time.png
```

Line styles encode Stage 2: solid = `Neural-RC`, dashed = `BeamSearch-RC`, dotted = `None`.

### End-to-end pipeline

```bash
./run_ablation.sh          # wait for SLURM array to complete
uv run python parse_logs.py
uv run python make_summary_table.py
uv run python plot_logs.py
```

Pre-computed tables and plots are already in `result/` from a prior run; re-run the scripts after new experiments to refresh them.

---

## Project layout

```
├── test.py                 # Ablation driver (reads STAGE_1 / STAGE_2)
├── TSPTester.py            # Stage 1 & 2 inference logic
├── train.py                # Stage 1 SL training
├── train_new.py            # Stage 2 RL fine-tuning
├── parse_logs.py           # Log → summary.csv
├── make_summary_table.py   # summary.csv → tables (CSV + LaTeX)
├── plot_logs.py            # summary.csv → figures
├── run_ablation.sh         # Submit SLURM ablation array
├── slurm_ablation.bash     # SLURM job array (9 configs)
├── slurm_train.bash        # SLURM Stage 2 training
└── result/                 # Logs, checkpoints, aggregated results
```

---

## Acknowledgements

Implementation is based on [LEHD](https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD). Synthetic benchmarks follow [INViT](https://github.com/Kasumigaoka-Utaha/INViT).

This ablation repository is a copy of the GELD tsp solvers repo with a few things added for ablation. If you use GELD cite them.
This repo is mostly not my code. 

### Citation

```bibtex
@ARTICLE{Xiao2025,
  author={Yubin Xiao and Di Wang and Rui Cao and Xuan Wu and Boyang Li and You Zhou},
  journal={Pattern Recognition},
  title={GELD: A unified neural model for efficiently solving traveling salesman problems across different scales},
  year={2026},
  volume={173},
  pages={1-15},
}
```
