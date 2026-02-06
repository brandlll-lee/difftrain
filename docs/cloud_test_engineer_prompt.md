# Cloud GPU Test Engineer Prompt (difftrain)

Role: Cloud Test Engineer (RTX 4090 or equivalent)

Mission: Execute the pytest-driven smoke test plan on a cloud GPU, record results, and report reproducible evidence.

This document is an execution runbook for the test engineer. Do not change code. Follow the rules below.

## Scope

- Run Text-to-Image MVP smoke tests on a GPU.
- Use pytest automation as the primary path.
- Fill in the checklist at `docs/mvp_smoke_checklist.md`.
- Collect and preserve reproducibility artifacts and logs.

## Non-Goals

- No refactors, no new features, no code edits.
- No dataset downloads beyond the dummy dataset or explicitly approved datasets.
- No model architecture changes.

## Required Inputs

- Repository: `difftrain`
- Config: `configs/smoke_text2image.yaml`
- Checklist: `docs/mvp_smoke_checklist.md`
- pytest markers: `slow`, `integration`, `cuda`, `text2image`

## Guardrails (Hard Rules)

- Only run commands approved by the ResearchEng lead.
- Do not modify tracked files unless explicitly instructed.
- If a command fails, capture the full error log and stop. Do not retry with ad-hoc changes.

## Execution Steps (Primary Path: pytest)

1. Install dependencies (with extras):
   - `python -m pip install -e "./difftrain[text2image,dev]"`

2. Run default unit tests (CPU, offline):
   - `pytest ./difftrain/tests -m "not slow and not integration" -v`

3. Run GPU training smoke (downloads model, 2 steps):
   - Optional helper for current shell:
     `source ./scripts/setup_runtime_env.sh`
   - Keep official Hugging Face endpoint by default; only fallback if unreachable:
     `curl -fsSIL https://huggingface.co >/dev/null || export HF_ENDPOINT=https://hf-mirror.com`
   - Ensure deterministic CUDA compatibility:
     `export CUBLAS_WORKSPACE_CONFIG=:4096:8`
   - `pytest ./difftrain/tests -m "slow and cuda and text2image" -v --run-slow --run-integration`

4. Fill the checklist:
   - `docs/mvp_smoke_checklist.md`

Notes:
If offline mode is enabled (`HF_HUB_OFFLINE=1` or `TRANSFORMERS_OFFLINE=1`), the GPU smoke will skip.

For CI/nightly parity, prefer:

- `pytest ./tests -m "slow and cuda and text2image" -v --run-slow --run-integration --basetemp ./_ci_tmp/pytest_tmp 2>&1 | tee ./_ci_tmp/logs/text2image_smoke.log`

Then verify:

- `./_ci_tmp/logs/text2image_smoke.log`
- `./_ci_tmp/pytest_tmp`

## Debug Path (Manual Scripts Only If pytest Fails)

Use this path only when the pytest smoke fails and you need to isolate the issue.

1. Create dummy ImageFolder dataset:
   - `python ./difftrain/scripts/make_dummy_imagefolder.py --output-dir ./data/smoke_text2image --num-images 4 --image-size 64`

2. Validate config + env snapshot:
   - `python ./difftrain/scripts/validate_config.py --config ./difftrain/configs/smoke_text2image.yaml --output-dir ./difftrain/_tmp_run --deterministic`

3. Run smoke training (2 steps):
   - `python ./difftrain/scripts/train_text_to_image.py --config ./difftrain/configs/smoke_text2image.yaml --output-dir ./difftrain/_tmp_run`

## What to Record

Fill the checklist (`docs/mvp_smoke_checklist.md`) with:

- Environment info (GPU model, driver, CUDA, Python, PyTorch)
- Command outputs summary (PASS/FAIL)
- Artifacts presence confirmation
- Any warnings, NaNs, or failures

Also capture:

- `nvidia-smi` output
- Full error logs if failures occur
- Smoke artifact paths used by CI parity run (`./_ci_tmp/logs/text2image_smoke.log`, `./_ci_tmp/pytest_tmp`)

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
