## Baseline Reproduction Checklist

This note captures the common patterns shared by the current end-to-end
pipelines (`delta_e2e`, `GNN-DSE/e2e_reimplement`, `progsg_e2e`). Treat it as
the contract for future baselines so results stay comparable.

### 1. CLI Contract
- Always accept an ID design root (`--design-root`) pointing to a structured
  ForgeHLS dump (e.g. `forgehls_lite_100designs`).
- Accept an optional OOD design root (`--ood-design-root`) with the same layout.
  When provided, evaluate on it after ID training/testing.
- Expose cache location switches (`--cache-dir`, `--rebuild-cache`) to avoid
  re-parsing `.autopilot/db/*.adb` graphs.
- Surface split control (`--train-ratio`, `--val-ratio`, `--seed`) plus limits
  like `--max-designs` / `--max-ood-designs` for quick smoke runs.
- Provide SwanLab toggles (`--no-swanlab`, `--swan-project`, `--swan-prefix`)
  so logging can be enabled/disabled without code edits.

### 2. Dataset Processing
- Materialise PyG graphs from `{design_root}/**/project/**/.autopilot/db/*.adb`
  and `csynth.xml`. Cache the processed tensors per dataset snapshot.
- Split the ID dataset into train/valid/test using the requested ratios; keep
  the full OOD dataset as a single evaluation loader.
- Record dataset sizes and split indices for logging/debugging.

### 3. Metrics & Reporting
- For every split (train/valid/test) and, if present, OOD:
  - Compute **MAE**, **RMSE**, **R²**, and **ulti-RMSE** (i.e. RMSE normalized by
    available resource counts: `dsp=9024`, `lut=1_303_680`, `ff=2_607_360`).
  - Optionally include **MAPE** or other study-specific stats, but never omit
    the core four metrics above.
- Track the best validation epoch; write a `history.json` capturing per-epoch
  train/valid losses and metrics.
- Dump predictions for each split to `<split>_predictions.csv` (and
  `ood_test_predictions.csv` when OOD is enabled).
- Print concise console summaries for ID test (and OOD test when provided)
  covering MAE/RMSE/R²/ulti-RMSE so runs leave a textual trace even without
  digging into artifact files.

### 4. SwanLab Logging
When SwanLab is available/enabled:
- Initialise a run using the CLI arguments as config (include dataset roots,
  cache paths, split ratios, resource caps, and runtime `run_id`).
- Log once:
  - Dataset sizes for train/valid/test/ood.
  - Training hyper-parameters (batch size, epochs, hidden dims, etc.).
- Per epoch:
  - Train RMSE/MSE.
  - Validation MAE/RMSE/R²/ulti-RMSE.
  - Best-epoch updates when validation improves.
- On completion:
  - Final train/valid/test/OOD metrics (MAE/RMSE/R²/ulti-RMSE).
  - Pointers to saved artifacts if helpful.

### 5. Artifacts Layout
For each run create a timestamped folder, e.g.
`<output_dir>/<metric>/<YYYYMMDD_HHMMSS>/`, containing at minimum:
- `best_model.pt`
- `config.json` (serialized CLI/config state)
- `history.json` (per-epoch log)
- `metrics.json` (summary of ID & OOD metrics, counts, artifact path)
- Prediction CSVs for every evaluated split

### 6. Implementation Skeleton
```text
1. Parse CLI → Build TrainConfig (design roots, cache, logging toggles).
2. Load/cache ID dataset → split → create DataLoaders.
3. Optional: load/cache OOD dataset → DataLoader.
4. Build model → run training with early stopping on validation RMSE.
5. Evaluate train/valid/test[/OOD] → compute MAE/RMSE/R²/ulti-RMSE.
6. Save artifacts & log through SwanLab/statistics module.
```

### 7. Example Commands
```bash
# ID-only quick check
python -m GNN-DSE.e2e_reimplement.run \
  --metric lut \
  --design-root ~/workspace/Huggingface/forgehls_lite_100designs \
  --epochs 200

# ID + OOD with capped samples and SwanLab disabled
python -m GNN-DSE.e2e_reimplement.run \
  --metric lut \
  --design-root ~/workspace/Huggingface/forgehls_lite_100designs \
  --ood-design-root ~/workspace/Huggingface/forgehls_benchmark \
  --max-designs 20 \
  --max-ood-designs 20 \
  --no-swanlab

# Code-only baseline (CodeT5 pooling)
python -m baseline.codeT5.run \
  --metric lut \
  --design-root ~/workspace/Huggingface/forgehls_lite_100designs \
  --ood-design-root ~/workspace/Huggingface/forgehls_benchmark \
  --max-designs 20 \
  --max-ood-designs 20 \
  --no-swanlab
```

Follow these points when standing up a new baseline to ensure consistent data
handling, metrics, and experiment tracking across projects.
