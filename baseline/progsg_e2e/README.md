# ProgSG Reproduction on ForgeHLS

This directory hosts a lightweight re-implementation of the ProgSG inference
pipeline tailored to the ForgeHLS dataset. It now mirrors the multimodal
setup from the original ProgSG work: we reuse the existing `delta_e2e` graph
decoding utilities, augment each design with CodeT5-based source-code
embeddings, and train a fused Transformer-style model to regress QoR metrics
(DSP, LUT, FF, latency).

## Model Overview

- **Graph construction**: each design’s HLS IR (`*.adb`) is parsed by
  `delta_e2e` utilities into a heterogeneous CDFG. For every node we encode
  category, opcode, bitwidth, resource buckets and pipeline attributes as
  categorical indices; edges carry type and back-edge flags. The dataset cache
  stores these integer tensors so repeated runs avoid reparsing.
- **Feature encoding**: the model embeds every categorical slot independently
  via learnable lookup tables and sums the embeddings to obtain dense node and
  edge representations. Pipeline metadata (e.g. pragma counts) is injected via
  an auxiliary MLP that is added to the pooled graph embedding.
- **Message passing**: stacked `TransformerConv` layers with multi-head
  attention and residual LayerNorm propagate information across the design
  graph. Global add pooling plus an MLP jointly regress DSP/LUT/FF/latency.
- **Source-code branch**: the ForgeHLS kernel source is tokenised with a
  cached CodeT5 encoder (`Salesforce/codet5-small` by default). Token
  embeddings are projected into the graph hidden space, optionally refined by
  a small Transformer encoder, and fused with the graph embedding via
  concatenation or residual addition.
- **Loss / metrics**: training minimises z-scored MSE, while evaluation reports
  MAE/RMSE/ulti-RMSE/R² in real units alongside per-design prediction dumps.

### Alignments with original ProgSG

- Inject multiple pipeline/context scalars (`pragma_count`, `has_pipeline`,
  `pipeline_region_count`, `avg_ii`, `max_pipe_depth`) via a small MLP into the
  pooled graph embedding.
- Optional node–token interaction: enable with `--code_node_interaction` to let
  graph nodes attend over code tokens before pooling.

For a reproducible end-to-end run on the 100-design split (with optional OOD
evaluation at `/home/user/zedongpeng/workspace/Huggingface/forgehls_benchmark`)
use the helper script:

```bash
cd progsg_e2e
./run_forgehls_100designs.sh
```

Environment variables such as `OUTPUT_DIR`, `EPOCHS`, `DEVICE`, `MAX_DESIGNS`
allow quick ablations; any extra CLI flags are forwarded to
`train_progsg_e2e.py`.

## Quick Start

```bash
source ../env/bin/activate
python train_progsg_e2e.py \
  --design_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_lite_100designs \
  --output_dir ./progsg_runs/forgehls_full \
  --cache_root ./progsg_cache \
  --epochs 100 \
  --batch_size 16 \
  --code_cache_device cuda
```

The script automatically splits the designs into 80%/10%/10% (train/valid/test)
and will optionally evaluate on an OOD directory when `--ood_design_base_dir`
is supplied.

Outputs include:
- `metrics_summary.json` for aggregate metrics (MAE, RMSE, ulti-RMSE, R²).
- `predictions_id_test.json` (and `predictions_ood_test.json` when applicable)
  containing per-design predictions vs. ground truth.

Set `--rebuild_cache` to force regeneration of graph caches when the dataset or
feature definitions change.

### Source-code modality specifics

- The first run downloads `Salesforce/codet5-small` and caches tokenised
  features under `progsg_cache/code_features_*`. Offline clusters can pre-stage
  the model (e.g. via `transformers-cli download`) and pass
  `CODE_LOCAL_FILES_ONLY=1`/`--code_local_files_only` to avoid network calls.
- Use `--code_encoder_name`/`CODE_ENCODER_NAME` to experiment with alternative
  checkpoints and `--code_max_length` to bound the token budget.
- The fusion head defaults to concatenation; select `--code_fusion_mode add`
  for residual blending or disable the modality entirely with `--disable_code`
  when reproducing the graph-only baseline.
 - Enable node–token interaction with `--code_node_interaction`.
