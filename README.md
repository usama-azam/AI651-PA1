# AI651 — Programming Assignment 1

Usama Azam · Roll no. 25280109 · LUMS, Fall 2026

Report: [`report/report.pdf`](report/report.pdf) (LaTeX source and numbered figure PDFs in `report/`).

## Task 1 — Forecasting across heterogeneous sensors

`Task1/Assignment1_solved.ipynb` is the executed notebook (`PA1_PRESET=full`, data seed 0, model seed 0, Kaggle T4 GPU). All notebook checks pass and Outputs 1.1–4.3 are included.

Implemented cells:
- `RawAttentionForecaster.forward`: embed → add positions → encoder blocks → LayerNorm → flatten → linear head.
- `SeriesDecomposition.forward`: centred moving average with replicate padding, returns `(remainder, trend)`.
- `aggregate_delays`: `z[t] = Σ_j w_j · v[(t − τ_j) mod L]`, per example, via an index tensor and `gather`.
- A validation-only deployment-choice rule (lowest validation RMSE; prefer a cheaper model within 2%), run before Output 4.3.

To rerun: keep `harness/` and `requirements.txt` next to the notebook, `pip install -r requirements.txt`, then run all cells. The first code cell sets `PA1_PRESET`; use `smoke` for a quick check and `full` for the reported results.

## Task 2 — Leaderboard challenge (Autoformer)

| File | Contents |
|---|---|
| `Task2/task2_v1_experiments.ipynb` | EDA, rolling-origin validation (26 weekly blocks), Autoformer implementation with unit checks, Experiment 1 (target transform × external-data ablation, 3 seeds, paired tests), Experiment 2 (occlusion), Experiment 3 (model size), first submission |
| `Task2/task2_v2_final.ipynb` | Final model: decisions on the 12 most recent weeks, 3 candidates × 5 seeds, blend/scale checked by leave-one-week-out CV, final 5-seed ensemble and the submitted forecast |
| `Task2/Data/` | Course data files |
| `Task2/results/` | Submitted forecasts (`leaderboard_paste_v*.txt`, 168 values from time_idx 43657 to 43824) and declared P, E (`submission_meta_v*.json`) |

**Final (v2) declared numbers:** P = 446,085 trainable parameters (5 × 89,217), E = 105 epochs (5 × 21).
Leaderboard: RMSE 81.1122, MAE 53.0061, sMAPE 49.94%, score 81.1656.

The Autoformer is written from scratch following Wu et al. (2021), §§3.1–3.2, and the structure of the reference implementation at https://github.com/thuml/Autoformer (series decomposition, Auto-Correlation with FFT delay scores and time-delay aggregation, encoder–decoder with trend accumulation). Differences: delays are chosen per example, and aggregation reads `v[(t − τ) mod L]` (the Task 1 convention). External covariates enter through a linear embedding in place of the timestamp embedding.

To reproduce: open a notebook with a GPU, keep the CSV files anywhere under the working directory (or `/kaggle/input` on Kaggle), and run all cells. Runtimes on a Kaggle T4: v1 ≈ 30 min, v2 ≈ 40 min. P and E are printed at the end of each notebook.

## Use of generative AI

Claude (Anthropic) was used as a learning partner and coding assistant; prompts, outputs and my edits are listed at the end of the report.
