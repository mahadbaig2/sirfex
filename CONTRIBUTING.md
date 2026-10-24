# Contributing to SIRFEX

SIRFEX is developed incrementally and verified locally before milestones are pushed. Contributions should preserve the local-first, CPU-first, reproducible design described in the technical PRD.

## Development setup

Use Python 3.12, create a virtual environment, and install the development and simulation extras:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,simulation]"
```

Run formatting, linting, and tests before opening a pull request:

```powershell
python -m ruff format --check src tests scripts
python -m ruff check src tests scripts
python -m pytest
```

Keep environment-specific physics in environment adapters and keep the evaluation runner environment-agnostic. Do not add cloud services, paid APIs, or GPU requirements without a documented architectural decision.

## License

By submitting a contribution, you agree that it is provided under the Apache License 2.0, as described in `LICENSE`. This project does not require a separate contributor license agreement.
