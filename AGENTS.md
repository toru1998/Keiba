# Repository Guidelines

## Project Structure & Module Organization
- Root: Python 3.12 project managed via `pyproject.toml` and `requirements.txt`.
- Source: `ScrapeData/modules/` organized by domain:
  - `preparing/` (scraping, drivers), `preprocessing/` (feature/data), `training/`, `simulation/`, `policies/`, `constants/`, `utils/`.
- Notebook: `ScrapeData/main.ipynb` for end‑to‑end exploration.
- Data: `data/` (e.g., `db/`, `html/`, `raw/`, `master/`) — paths centralized in `modules/constants/_local_paths.py`.

## Build, Test, and Development Commands
- Create venv: `python -m venv .venv && source .venv/bin/activate` (Windows: `Scripts\activate`).
- Install deps: `pip install -r requirements.txt` (or `uv sync` if using `uv`).
- Launch notebooks: `jupyter lab` in repo root.
- Run a module: `python -m ScrapeData.modules.preparing._prepare_chrome_driver` (adapt module path as needed).

## Coding Style & Naming Conventions
- Python: PEP 8, 4‑space indentation, 120‑char soft line limit.
- Types: add type hints for public functions and dataclasses where practical.
- Names: modules/functions `snake_case`, classes `CamelCase`. Internal/experimental modules are prefixed with `_` (follow existing pattern).
- Imports: prefer absolute package paths under `ScrapeData.modules`.

## Testing Guidelines
- Location: place tests in `tests/` mirroring package paths.
- Naming: files `test_*.py`; functions `test_*`.
- Run: `pytest -q` (add `pytest` to dev env if not installed).
- Coverage: target critical paths (scrapers, processors, policies); prioritize pure functions and I/O boundaries.

## Commit & Pull Request Guidelines
- Branches: `feat/<scope>`, `fix/<scope>`, `chore/<scope>`.
- Commits: concise imperative subject, e.g., `feat(preprocessing): add race info merger`.
- PRs: include purpose, summary of changes, usage notes, and screenshots/logs when UI or plots change. Link related issues.
- Checks: ensure notebooks run top‑to‑bottom, no hardcoded local paths (use `LocalPaths`), and no large data files committed.

## Security & Configuration Tips
- Selenium/Chrome: prefer `webdriver-manager`; avoid embedding credentials.
- Data paths: rely on `LocalPaths` for portability; don’t mutate it at runtime.
- Secrets/artifacts: keep out of VCS; use `.gitignore` and `data/` for generated assets.
