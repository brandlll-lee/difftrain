# difftrain

Research-first diffusion training toolkit focused on reproducible training loops and strategy experiments.

Status: MVP runnable (text-to-image only). Not production-ready.

**What It Is**
difftrain is a minimal, composable training skeleton for diffusion models. The long-term direction is a
multi-modal diffusion training platform (text, image, video, audio) with clear contracts, reproducibility
artifacts, and extensible training strategies. The current MVP focuses on a single, stable vertical:
text-to-image latent diffusion with SD-style UNet and CLIP conditioning.

**Design Goals**
- Minimal end-to-end training loop that actually runs.
- Reproducibility artifacts for every run (config, env, manifest).
- Clean extension points for strategies, backbones, and modalities.
- Low dependency core with optional extras for heavy stacks.

**Current Implementation Status**
| Area | Status | Notes |
| --- | --- | --- |
| Text-to-image training loop | Implemented | `training/train_step.py`, `training/runner.py` |
| DREAM (epsilon + v_prediction) | Implemented | `training/dream.py` |
| VAE latent encoding | Implemented | `training/train_step.py` |
| CLIP text conditioning | Implemented | `scripts/train_text_to_image.py` |
| ImageFolder dataset pipeline | Implemented | Requires `metadata.jsonl` |
| AMP (fp16/bf16) | Implemented | `train.precision` |
| Resume-from-checkpoint | Implemented | UNet + optimizer + global_step + scaler |
| Repro artifacts | Implemented | `env.json`, `config.resolved.yaml`, `run_manifest.json` |
| Smoke config + dummy data | Implemented | `configs/smoke_text2image.yaml` |
| Unit tests | Implemented | `tests/` (pytest markers for unit/integration/slow) |
| Runner/train-step branch tests | Implemented | Covers `sample`, `dream_training=True`, and key error branches |
| Benchmarks | Implemented | `benchmarks/benchmark_train_step.py` |
| report_to / eval_every / save_every | Partial | Config fields present, not wired |
| Loss variants (p2/snr/huber) | Partial | Config fields present, MSE only |
| Full resume (RNG + dataloader) | Not yet | Resume does not restore order |
| DDP/FSDP/ZeRO | Not yet | Single-process only |
| EMA / eval / sampling | Not yet | Hooks not implemented |
| Pixel-space / DiT / multimodal | Not yet | MVP is text-to-image only |

**Architecture Overview**
1. VAE encodes images into latents.
2. Timesteps and noise are sampled.
3. Scheduler adds noise to latents.
4. Text encoder produces conditioning states.
5. Target is built based on prediction type.
6. Optional DREAM correction updates latents and target.
7. UNet predicts noise or velocity.
8. Loss is computed and optimized.

Key files:
- `src/difftrain/training/train_step.py`
- `src/difftrain/training/runner.py`
- `src/difftrain/training/dream.py`
- `src/difftrain/config.py`

**Project Layout**
- `configs/` experiment configs
- `src/difftrain/` library code
- `scripts/` training and utility scripts
- `tests/` unit tests
- `benchmarks/` micro-benchmarks
- `docs/` specs and runbooks

**Quickstart (Text-to-Image MVP)**
1. Install extras:
   `python -m pip install -e "./difftrain[text2image]"`
2. Create dummy dataset:
   `python ./difftrain/scripts/make_dummy_imagefolder.py --output-dir ./data/smoke_text2image --num-images 4 --image-size 64`
3. Run smoke training:
   `python ./difftrain/scripts/train_text_to_image.py --config ./difftrain/configs/smoke_text2image.yaml --output-dir ./difftrain/_tmp_run`

Dataset note:
- The ImageFolder loader expects `metadata.jsonl` with a `file_name` field and a caption field.
- The default caption column is `text` (see `data.caption_column` in the config).

**Precision and Resume**
- Set `train.precision` to `fp32`, `fp16`, or `bf16`.
- Resume example:
  `python ./difftrain/scripts/train_text_to_image.py --config ./difftrain/configs/smoke_text2image.yaml --output-dir ./difftrain/_tmp_run --resume-from ./difftrain/_tmp_run/checkpoint-2.pt`
- Resume restores model/optimizer/scaler state and global step, but not dataloader order.

**Reproducibility Artifacts**
- `env.json` environment snapshot
- `config.resolved.yaml` resolved config
- `run_manifest.json` git info + dependency versions

Validate snapshot pipeline without training:
`python ./difftrain/scripts/validate_config.py --config ./difftrain/configs/smoke_text2image.yaml --output-dir ./difftrain/_tmp_run --deterministic`

**Tests and Benchmarks**
- Default unit tests (CPU, offline):
  `python -m pip install -e "./difftrain[dev]" && pytest ./difftrain/tests -m "not slow and not integration" -v`
- GPU training smoke (downloads model, 2 steps):
  `python -m pip install -e "./difftrain[text2image,dev]" && pytest ./difftrain/tests -m "slow and cuda and text2image" -v --run-slow --run-integration`
- Train-step micro-benchmark:
  `python ./difftrain/benchmarks/benchmark_train_step.py`

GPU smoke notes:
- Session helper (recommended): `source ./scripts/setup_runtime_env.sh`
- Deterministic CUDA runs require `CUBLAS_WORKSPACE_CONFIG` (the code now defaults to `:4096:8` when deterministic mode is enabled and CUDA is available).
- Keep official `huggingface.co` as default. Only set a mirror endpoint when official access fails:
  `curl -fsSIL https://huggingface.co >/dev/null || export HF_ENDPOINT=https://hf-mirror.com`

**CI Lanes**
- `tests` (push + pull_request): CPU matrix on Python 3.10/3.11; runs `pytest ./tests -v` and `ruff check .`.
- `text2image-smoke` (nightly + manual dispatch): runs GPU smoke on self-hosted runner labels `self-hosted,linux,x64,gpu`.
- Nightly lane stores smoke artifacts at:
  - log: `./_ci_tmp/logs/text2image_smoke.log`
  - pytest temp/artifacts: `./_ci_tmp/pytest_tmp`
  - uploaded artifact: `text2image-smoke-${{ github.run_id }}`

**Docs and Runbooks**
- MVP spec: `docs/mvp_spec.md`
- Smoke checklist (pytest-driven): `docs/mvp_smoke_checklist.md`
- Cloud test engineer prompt: `docs/cloud_test_engineer_prompt.md`

**Roadmap (Near-Term)**
- Implement loss variants (P2, SNR weighting, Huber)
- Add full resume (RNG + dataloader state)
- Add evaluation and sampling hooks
- Add distributed training primitives (DDP/FSDP)

**Contributing**
See `CONTRIBUTING.md` for development workflow and guidelines.

**Code of Conduct**
See `CODE_OF_CONDUCT.md`.

**License**
Apache-2.0. See `LICENSE`.
