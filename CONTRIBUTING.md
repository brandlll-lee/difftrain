# Contributing to difftrain

Thank you for considering a contribution. This project is research-first and
prioritizes correctness, reproducibility, and clear contracts over feature volume.

## Quick Setup

1. Create a virtual environment.
2. Install editable deps:
   - `python -m pip install -e "./difftrain[dev]"`
3. Run tests:
   - Default unit tests (CPU, offline):
     `pytest ./difftrain/tests -m "not slow and not integration" -v`
   - Optional GPU smoke (downloads model, 2 steps):
     `pytest ./difftrain/tests -m "slow and cuda and text2image" -v --run-slow --run-integration`

## CI Lanes

- `tests` lane runs on push/PR with CPU runners and executes `pytest ./tests -v` and `ruff check .`.
- `text2image-smoke` lane runs nightly (and by manual dispatch) on self-hosted GPU runners.
- Smoke lane command:
  `pytest ./tests -m "slow and cuda and text2image" -v --run-slow --run-integration --basetemp ./_ci_tmp/pytest_tmp`
- Smoke lane captures logs and artifacts:
  - `./_ci_tmp/logs/text2image_smoke.log`
  - `./_ci_tmp/pytest_tmp`

If your organization uses different self-hosted runner labels, update `runs-on` in `.github/workflows/ci.yml`.

## Pull Request Guidelines

- Keep changes focused and small when possible.
- Add or update unit tests for new behaviors.
- Update docs if you change user-facing behavior.
- Prefer clear, explicit code over clever code.
- Run formatting/linting if applicable.

## Issues

When reporting a bug, include:

- OS + Python version
- PyTorch version
- Repro steps and minimal config
- Full error logs

## Design Principles

- Reproducibility first: every run should emit env/config/manifest artifacts.
- Extension over replacement: add adapters/interfaces rather than breaking MVP contracts.
- Minimal dependencies in core; use optional extras for heavy stacks.
