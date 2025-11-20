# CodeT5 Baseline

This baseline mirrors the shared checklist in `baseline/README.md` while
focusing purely on source-code signals (no graphs). It uses CodeT5 to encode the
design source and regresses QoR targets with a shallow MLP head.

## Quick Start

```bash
python -m baseline.codeT5.run --metric lut
```

Full options:

```bash
python -m baseline.codeT5.run --help
```

For a lightweight smoke test without SwanLab or GPU:

```bash
bash baseline/codeT5/run_quicktest.sh
```
