# Contributing to difftrain

Thank you for considering a contribution. This project is research-first and
prioritizes correctness, reproducibility, and clear contracts over feature volume.

## Quick Setup

1. Create a virtual environment.
2. Install editable deps:
   - `python -m pip install -e "./difftrain[dev]"`
3. Run tests:
   - `pytest ./difftrain/tests -v`

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
