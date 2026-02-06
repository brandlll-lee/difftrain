# difftrain MVP Smoke Test Checklist (Text-to-Image)

This checklist verifies the Text-to-Image MVP training path (SD-style UNet + VAE latents + CLIP text encoder).
The canonical path is pytest-driven automation. Manual scripts are for debugging only.

**Primary path**: `pytest` (automated checks and artifacts validation)  
**Secondary path**: manual scripts (only when debugging failures)

## Run Metadata (Fill In)

| Field | Value |
| --- | --- |
| Date | |
| Operator | |
| Location | |
| Machine | |
| GPU (model, VRAM) | |
| Driver version | |
| CUDA version | |
| Python version | |
| PyTorch version | |
| diffusers version | |
| transformers version | |
| datasets version | |
| torchvision version | |
| OS | |
| Repo path | |
| Git commit | |
| Working tree clean (Y/N) | |

## Preconditions (Required)

| Step | Command | Expected | Result (PASS/FAIL/NA) | Notes |
| --- | --- | --- | --- | --- |
| Install extras | `python -m pip install -e "./difftrain[text2image,dev]"` | No missing deps | | |
| Confirm repo root | `pwd` | Matches repo path above | | |
| CUDA availability | `python -c "import torch; print(torch.cuda.is_available())"` | `True` | | |

Notes:
`configs/smoke_text2image.yaml` uses `data.data_root: data/smoke_text2image`.  
The pytest smoke test will create a dummy dataset automatically at a temp path.

If offline mode is enabled, the GPU smoke test will be skipped.
Check `HF_HUB_OFFLINE` or `TRANSFORMERS_OFFLINE`.

For deterministic CUDA runs, ensure:
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` (or `:16:8`).

Endpoint policy:
- Default to official `huggingface.co`.
- Fallback to mirror only when official endpoint is unreachable:
  `curl -fsSIL https://huggingface.co >/dev/null || export HF_ENDPOINT=https://hf-mirror.com`
- Optional helper (sets both policies for current shell):
  `source ./scripts/setup_runtime_env.sh`

## Primary Path: Pytest Automation (Required)

### A. Default unit tests (CPU, offline)

| Step | Command | Expected | Result (PASS/FAIL/NA) | Notes |
| --- | --- | --- | --- | --- |
| Run unit tests | `pytest ./difftrain/tests -m "not slow and not integration" -v` | All pass | | |

### B. GPU training smoke (downloads model, 2 steps)

| Step | Command | Expected | Result (PASS/FAIL/NA) | Notes |
| --- | --- | --- | --- | --- |
| Run GPU smoke | `pytest ./difftrain/tests -m "slow and cuda and text2image" -v --run-slow --run-integration` | Completes 2 steps without crash | | |

Automated validations included in the GPU smoke:
- Dummy dataset creation (ImageFolder with `metadata.jsonl`)
- Model download `hf-internal-testing/tiny-stable-diffusion-pipe`
- Training for 2 steps
- Repro artifacts: `env.json`, `config.resolved.yaml`, `run_manifest.json`
- Checkpoints: `checkpoint-1.pt`, `checkpoint-2.pt`

Note: pytest writes artifacts under a temporary directory. Capture the output path from pytest logs.

### C. CI Nightly Lane (Reference)

The repository CI includes a dedicated nightly GPU smoke lane (`text2image-smoke`).
It also supports manual dispatch from GitHub Actions.

| Step | Command / Location | Expected | Result (PASS/FAIL/NA) | Notes |
| --- | --- | --- | --- | --- |
| Smoke command in CI | `pytest ./tests -m "slow and cuda and text2image" -v --run-slow --run-integration --basetemp ./_ci_tmp/pytest_tmp` | Completes without crash | | |
| Smoke log path | `./_ci_tmp/logs/text2image_smoke.log` | File exists (even on failure) | | |
| Pytest temp path | `./_ci_tmp/pytest_tmp` | Directory exists | | |
| Uploaded artifact | `text2image-smoke-${{ github.run_id }}` | Visible in Actions run artifacts | | |

## Optional: Manual Script Path (Debug Only)

Use these only if pytest fails and you need to isolate the issue.

| Step | Command | Expected | Result (PASS/FAIL/NA) | Notes |
| --- | --- | --- | --- | --- |
| Create dummy dataset | `python ./difftrain/scripts/make_dummy_imagefolder.py --output-dir ./data/smoke_text2image --num-images 4 --image-size 64` | Folder with `metadata.jsonl` and 4 images | | |
| Validate config + env snapshot | `python ./difftrain/scripts/validate_config.py --config ./difftrain/configs/smoke_text2image.yaml --output-dir ./difftrain/_tmp_run --deterministic` | Writes repro artifacts | | |
| Run training smoke | `python ./difftrain/scripts/train_text_to_image.py --config ./difftrain/configs/smoke_text2image.yaml --output-dir ./difftrain/_tmp_run` | Completes 2 steps without crash | | |

## Optional: AMP Manual Smoke (Debug Only)

| Step | Command | Expected | Result (PASS/FAIL/NA) | Notes |
| --- | --- | --- | --- | --- |
| fp16 smoke | `python ./difftrain/scripts/train_text_to_image.py --config ./difftrain/configs/smoke_text2image.yaml --output-dir ./difftrain/_tmp_run_fp16` | Completes 2 steps without crash | | |
| bf16 smoke | `python ./difftrain/scripts/train_text_to_image.py --config ./difftrain/configs/smoke_text2image.yaml --output-dir ./difftrain/_tmp_run_bf16` | Completes 2 steps without crash | | |

Note: set `train.precision` in the config to `fp16` or `bf16` for these runs.

## Issues / Follow-ups

| Item | Details |
| --- | --- |
| Issues encountered | |
| Proposed fixes | |
| Next actions | |
