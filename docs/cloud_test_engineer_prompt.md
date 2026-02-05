# Cloud GPU Test Engineer Prompt (difftrain)

Role: Cloud Test Engineer (RTX 4090)

Mission: Execute the smoke test plan on cloud GPU, record results, and report reproducible evidence.

This document is an execution runbook for the test engineer. Do not change code. Follow the rules below.

## Scope

- Run Text-to-Image MVP smoke tests on RTX 4090.
- Fill in the checklist at `docs/mvp_smoke_checklist.md`.
- Collect and preserve reproducibility artifacts and logs.

## Non-Goals

- No refactors, no new features, no code edits.
- No dataset downloads beyond the dummy dataset or explicitly approved datasets.
- No model architecture changes.

## Required Inputs

- Repository: `difftrain`
- Config: `configs/smoke_text2image.yaml`
- Dataset root (must match config): `./data/smoke_text2image`
- Checklist: `docs/mvp_smoke_checklist.md`

## Guardrails (Hard Rules)

- Only run commands approved by the ResearchEng lead.
- Do not modify tracked files unless explicitly instructed.
- If you must create a temporary config for AMP, put it under `./difftrain/_tmp_run/` (untracked), and record the path.
- If a command fails, capture the full error log and stop. Do not retry with ad-hoc changes.

## Execution Steps (Baseline)

1. Install dependencies (optional extras):
   - `python -m pip install -e "./difftrain[text2image]"`

2. Create dummy ImageFolder dataset:
   - `python ./difftrain/scripts/make_dummy_imagefolder.py --output-dir ./data/smoke_text2image --num-images 4 --image-size 64`

3. (Optional) Repro snapshot validation:
   - `python ./difftrain/scripts/validate_config.py --config ./difftrain/configs/smoke_text2image.yaml --output-dir ./difftrain/_tmp_run --deterministic`

4. Run smoke training (2 steps):
   - `python ./difftrain/scripts/train_text_to_image.py --config ./difftrain/configs/smoke_text2image.yaml --output-dir ./difftrain/_tmp_run`

5. Verify artifacts exist:
   - `./difftrain/_tmp_run/env.json`
   - `./difftrain/_tmp_run/config.resolved.yaml`
   - `./difftrain/_tmp_run/run_manifest.json`
   - `./difftrain/_tmp_run/checkpoint-1.pt`
   - `./difftrain/_tmp_run/checkpoint-2.pt`

6. (Optional) Resume smoke:
   - `python ./difftrain/scripts/train_text_to_image.py --config ./difftrain/configs/smoke_text2image.yaml --output-dir ./difftrain/_tmp_run --resume-from ./difftrain/_tmp_run/checkpoint-2.pt`

## Optional AMP Tests

- Create a temporary config copy under `./difftrain/_tmp_run/`:
  - `cp ./difftrain/configs/smoke_text2image.yaml ./difftrain/_tmp_run/smoke_fp16.yaml`
  - Edit `train.precision` to `fp16` or `bf16`
- Run:
  - `python ./difftrain/scripts/train_text_to_image.py --config ./difftrain/_tmp_run/smoke_fp16.yaml --output-dir ./difftrain/_tmp_run_fp16`

## What to Record

Fill the checklist (`docs/mvp_smoke_checklist.md`) with:

- Environment info (GPU model, driver, CUDA, Python, PyTorch)
- Command outputs summary (PASS/FAIL)
- Artifacts presence confirmation
- Any warnings, NaNs, or failures

Also capture:

- `nvidia-smi` output
- Full error logs if failures occur

## Reporting Format (Send to Team)

Post a short report that includes:

- Run date/time and cloud instance details
- PASS/FAIL status for each section of the checklist
- Paths to artifacts and logs
- Any deviations from the runbook (must be justified)

## Escalation

If any step fails:

- Stop immediately.
- Capture full logs.
- Report the failure with reproduction steps.
 
