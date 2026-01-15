# important
日本語で回答してください

# AGENTS.md - Keiba Project

This document provides guidelines for AI coding agents working on this Japanese horse racing (Keiba) data analysis project.

## Project Overview

A Python project for parsing, processing, and analyzing Japanese horse racing data from netkeiba.com HTML files. Uses pandas for data manipulation and LightGBM for predictive modeling.

**Main Technologies:** Python 3.11, pandas, BeautifulSoup/lxml, LightGBM, scikit-learn

## Directory Structure

```
Keiba/
├── src/                    # Python source modules
│   ├── html2pandasdf.py    # HTML parsing to DataFrames
│   └── preprocessing_race_results.py  # Data preprocessing
├── notebook/               # Jupyter notebooks for analysis/modeling
├── data/                   # Data storage
│   ├── html/race/          # Source HTML files (gzipped)
│   └── *.pkl               # Processed pickle files
├── csv/                    # CSV output files
├── main.py                 # Entry point
├── pyproject.toml          # Project dependencies (uv)
└── uv.lock                 # Lock file
```

## Build/Run Commands

### Environment Setup

```bash
# Install dependencies using uv
uv pip install -e .

# Or sync from lock file
uv sync
```

### Running Scripts

```bash
# Run main entry point
uv run python main.py

# Process HTML files to pickle
uv run python src/html2pandasdf.py

# Run a specific Python file
uv run python <path/to/script.py>
```

### Jupyter Notebooks

```bash
# Start Jupyter for notebook work
uv run jupyter notebook

# Run notebook from command line
uv run jupyter nbconvert --to notebook --execute notebook/<name>.ipynb
```

### Testing

**Note:** This project currently has no formal test suite (no pytest, unittest).

```bash
# If tests are added in the future, use:
uv run pytest                           # Run all tests
uv run pytest tests/test_file.py        # Run single test file
uv run pytest tests/test_file.py::test_name  # Run single test
uv run pytest -v                        # Verbose output
uv run pytest -x                        # Stop on first failure
```

### Linting/Formatting

**Note:** No linting configuration exists yet. If adding linting:

```bash
# Recommended: Add ruff to pyproject.toml, then:
uv run ruff check .                     # Lint
uv run ruff format .                    # Format
uv run ruff check --fix .               # Auto-fix
```

## Code Style Guidelines

### Imports

Order imports as follows, separated by blank lines:

1. Standard library imports
2. Third-party imports
3. Local/project imports

```python
import gzip
from pathlib import Path
from io import StringIO

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.html2pandasdf import parse_race_html
```

### Type Hints

Use type hints for function signatures:

```python
def parse_race_html(html_content: str, race_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """HTMLからレース結果、払い戻し、ラップタイムを抽出"""
    ...
```

### Naming Conventions

| Element       | Style         | Example                    |
|---------------|---------------|----------------------------|
| Functions     | snake_case    | `load_all_race_html()`     |
| Variables     | snake_case    | `race_results`, `html_file`|
| Constants     | UPPER_SNAKE   | `DATA_DIR`, `MAX_RETRIES`  |
| Classes       | PascalCase    | `RaceParser`, `DataLoader` |
| Files         | snake_case    | `html2pandasdf.py`         |

### Docstrings and Comments

- Use Japanese for docstrings and comments (this is a Japanese domain project)
- Keep docstrings concise but descriptive

```python
def parse_race_html(html_content: str, race_id: str) -> tuple[...] | None:
    """HTMLからレース結果、払い戻し、ラップタイムを抽出してDataFrameに変換"""
```

### Error Handling

- Use specific exception types, not bare `except:`
- Return `None` or empty DataFrames for graceful degradation
- Log errors with context (file path, race_id, etc.)

```python
try:
    dfs = pd.read_html(StringIO(html_content))
except (ValueError, IndexError):
    return None
except Exception as e:
    print(f"Error reading {html_file}: {e}")
    return None
```

### DataFrame Patterns

- Use `race_id` as the standard identifier column
- Set `race_id` as index after concatenation
- Return empty DataFrames rather than None for optional data

```python
combined_results = pd.concat(results_list, ignore_index=True)
combined_results.set_index('race_id', inplace=True)
```

### Entry Point Pattern

Always use the `if __name__ == "__main__":` guard:

```python
if __name__ == "__main__":
    results_df, payouts_df, laps_df = load_all_race_html()
    results_df.to_pickle("data/race_results.pkl")
```

## Data Conventions

### HTML Table Structure (from netkeiba.com)

| Index | Content              | Notes                    |
|-------|----------------------|--------------------------|
| 0     | Race results         | Main results table       |
| 1-2   | Payout information   | Betting payouts          |
| 3     | Track conditions     | Premium only (ignore)    |
| 4     | Corner positions     | Horse positions by corner|
| 5     | Lap times            | Race pace data           |

### Output Files

- Use pickle (`.pkl`) for DataFrame storage
- Use descriptive names: `race_results.pkl`, `race_payouts.pkl`, `race_laps.pkl`
- Store in `data/` directory

### Race ID Format

Race IDs follow netkeiba's format: `YYYYPPTTRRNN`
- YYYY: Year
- PP: Place code
- TT: Kai (meeting number)
- RR: Day
- NN: Race number

## Notebook Guidelines

- Use notebooks in `notebook/` for exploration and model development
- Include markdown cells explaining analysis steps
- Use `japanize-matplotlib` for Japanese text in plots
- Reference pickled data with relative paths: `../data/race_results.pkl`

## Key Domain Terms (Japanese)

| Japanese | English           | Column Example |
|----------|-------------------|----------------|
| 着順     | Finishing position| 着 順          |
| 枠番     | Post position     | 枠 番          |
| 馬番     | Horse number      | 馬 番          |
| 性齢     | Sex and age       | 性齢           |
| 斤量     | Weight carried    | 斤量           |
| 単勝     | Win odds          | 単勝           |
| 人気     | Popularity rank   | 人 気          |
| 馬体重   | Horse weight      | 馬体重         |

## Adding New Features

1. Add data extraction logic to `src/html2pandasdf.py`
2. Add preprocessing to `src/preprocessing_race_results.py`
3. Prototype in a notebook first, then refactor to modules
4. Update pickle outputs if schema changes
