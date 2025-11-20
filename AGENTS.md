# Repository Guidelines

## Project Structure & Module Organization
- `GNN/src/` hosts models and the training/inference entry points; keep new variants and helpers inside this tree.
- Dataset builders live in `Graphs/`, with caches staged in `graph_cache/` and `GNN/dataset/`; never commit extracted Hugging Face archives.
- Experiment shell scripts sit under `GNN/`, while root utilities like `test_pyg.py` or `install_pyg.sh` provide environment checks—extend them instead of duplicating logic.

## Build, Test, and Development Commands
- `python Graphs/process_real_case_graph_PolyBench.py` builds PolyBench graphs; move the emitted `dataset/` into `GNN/dataset/`.
- `python GNN/src/check_dataset_valid.py --dataset_name <name>` validates graph statistics before training.
- `bash GNN/train.sh` (or a variant script) launches the experiment parameters defined in `GNN/src/make_master_file.py`.
- `bash GNN/inference_on_PolyBench.sh` reports regression metrics—capture the table for review.
- `python test_pyg.py` sanity-checks torch-geometric installs after environment changes.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and keep lines near the 120-character soft limit used in `train.py`.
- Use descriptive snake_case for functions, files, and script names (`train_on_forgehls-lite_4090.sh` is the pattern).
- Centralize configuration near the top of a module, reuse existing `argparse` setups, and comment only when control flow is non-obvious.

## Testing Guidelines
- Add new tests under `tests/` or `GNN/tests/` and run them with `python -m pytest`.
- Rerun `check_dataset_valid.py` whenever you change preprocessing, encoders, or cached datasets.
- Before sharing results, rerun the matching inference script and include the metric table or summary in the PR notes.

## Commit & Pull Request Guidelines
- Follow the `type: short summary` convention seen in history (`feat:`, `fix:`) and keep subjects under 72 characters.
- PRs should link related issues, state the dataset snapshot, and note key hardware assumptions (GPU, batch size, epochs).
- Summarize metric deltas or attach tables; store heavyweight artifacts externally and reference them in the description.

## Security & Environment Notes
- Keep machine-specific paths or secrets out of commits; expose them through CLI flags or config files.
- Ensure `.gitignore` covers caches (`GNN/saves/`, tarballs) and document CUDA/PyTorch versions when scripts rely on a specific stack.
