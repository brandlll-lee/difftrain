# difftrain MVP Smoke Test Checklist (Text-to-Image)

This checklist verifies the Text-to-Image MVP training path (SD-style UNet + VAE latents + CLIP text encoder).
Fill in the results after running on your GPU environment.

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

## Preflight (Required)

| Step | Command | Expected | Result (PASS/FAIL/NA) | Notes |
| --- | --- | --- | --- | --- |
| Install editable with extras | `python -m pip install -e "./difftrain[text2image]"` | No missing deps | | |
| Confirm repo root | `pwd` | Matches repo path above | | |

## Dataset (Required)

Important: `configs/smoke_text2image.yaml` uses `data.data_root: data/smoke_text2image`.  
Make sure the dataset path you generate matches the config (or adjust the config and record the change).

| Step | Command | Expected | Result (PASS/FAIL/NA) | Notes |
| --- | --- | --- | --- | --- |
| Create dummy dataset | `python ./difftrain/scripts/make_dummy_imagefolder.py --output-dir ./data/smoke_text2image --num-images 4 --image-size 64` | Folder with `metadata.jsonl` and 4 images | | |
| Verify dataset exists | `ls ./data/smoke_text2image` | Shows images + `metadata.jsonl` | | |

## Repro Snapshot (Recommended)

| Step | Command | Expected | Result (PASS/FAIL/NA) | Notes |
| --- | --- | --- | --- | --- |
| Validate config + env snapshot | `python ./difftrain/scripts/validate_config.py --config ./difftrain/configs/smoke_text2image.yaml --output-dir ./difftrain/_tmp_run --deterministic` | Writes `env.json`, `config.resolved.yaml`, `run_manifest.json` | | |

## Training Smoke (Required)

| Step | Command | Expected | Result (PASS/FAIL/NA) | Notes |
| --- | --- | --- | --- | --- |
| Run training smoke | `python ./difftrain/scripts/train_text_to_image.py --config ./difftrain/configs/smoke_text2image.yaml --output-dir ./difftrain/_tmp_run` | Completes 2 steps without crash | | |

## Artifacts Check (Required)

| Step | Path | Expected | Result (PASS/FAIL/NA) | Notes |
| --- | --- | --- | --- | --- |
| Env snapshot | `./difftrain/_tmp_run/env.json` | File exists | | |
| Resolved config | `./difftrain/_tmp_run/config.resolved.yaml` | File exists | | |
| Run manifest | `./difftrain/_tmp_run/run_manifest.json` | File exists | | |
| Checkpoint step 1 | `./difftrain/_tmp_run/checkpoint-1.pt` | File exists | | |
| Checkpoint step 2 | `./difftrain/_tmp_run/checkpoint-2.pt` | File exists | | |

## Loss Sanity (Required)

| Field | Value |
| --- | --- |
| Last reported loss | |
| Any NaNs? (Y/N) | |
| Any runtime warnings? | |

## Optional: AMP Smoke

| Step | Command | Expected | Result (PASS/FAIL/NA) | Notes |
| --- | --- | --- | --- | --- |
| fp16 smoke | `python ./difftrain/scripts/train_text_to_image.py --config ./difftrain/configs/smoke_text2image.yaml --output-dir ./difftrain/_tmp_run_fp16` | Completes 2 steps without crash | | |
| bf16 smoke | `python ./difftrain/scripts/train_text_to_image.py --config ./difftrain/configs/smoke_text2image.yaml --output-dir ./difftrain/_tmp_run_bf16` | Completes 2 steps without crash | | |

Note: set `train.precision` in the config to `fp16` or `bf16` for these runs.

## Optional: Resume-from-checkpoint

| Step | Command | Expected | Result (PASS/FAIL/NA) | Notes |
| --- | --- | --- | --- | --- |
| Resume from checkpoint | `python ./difftrain/scripts/train_text_to_image.py --config ./difftrain/configs/smoke_text2image.yaml --output-dir ./difftrain/_tmp_run --resume-from ./difftrain/_tmp_run/checkpoint-2.pt` | Continues without crash | | |

Note: resume restores model/optimizer/scaler state and global step, but not dataloader order.

## Optional: Repro Smoke Script

| Step | Command | Expected | Result (PASS/FAIL/NA) | Notes |
| --- | --- | --- | --- | --- |
| Check repro artifacts | `python ./difftrain/scripts/smoke_repro.py --output-dir ./difftrain/_tmp_run` | Prints `smoke_repro: OK` | | |

## Issues / Follow-ups

| Item | Details |
| --- | --- |
| Issues encountered | |
| Proposed fixes | |
| Next actions | |
