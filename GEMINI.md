# Project Overview

This is a Python project for analyzing Japanese horse racing (Keiba) data. The main goal is to parse race result data from local HTML files, process it using the pandas library, and consolidate it into a single data file for further analysis.

The project is structured to:
1.  Read compressed HTML files (`.html.gz`) from the `data/html/race` directory.
2.  Parse each HTML file to extract the main race result table.
3.  Append a `race_id` to each race's data, derived from the filename.
4.  Concatenate the data from all files into a single pandas DataFrame.
5.  Save the final DataFrame to a pickle file (`data/html_data.pkl`).

**Main Technologies:**
*   Python
*   Pandas
*   Beautiful Soup (for HTML parsing, via pandas)
*   lxml

# Building and Running

## 1. Setup

It is recommended to use a virtual environment. Dependencies are managed with `uv` and listed in `pyproject.toml`.

```bash
# Install dependencies
uv pip install -e .
```

## 2. Data Processing

The main script for processing the HTML files is `html2pandasdf.py`. To run it, execute the following command from the project root:

```bash
python html2pandasdf.py
```

This will read the HTML files located in `data/html/race`, process them, and save the output to `data/html_data.pkl`.

# Development Conventions

*   **Data Source:** The primary data source is HTML files, which are expected to be stored as gzipped archives in the `data/html/race/` directory.
*   **Core Logic:** The main data processing logic resides in `html2pandasdf.py`.
*   **Output:** The standard output for processed data is a pandas DataFrame serialized to a pickle file.
*   **Notebooks:** The presence of `test.ipynb` and `ipykernel` suggests that Jupyter notebooks are used for testing, exploration, and analysis of the processed data.
