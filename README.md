# data-sweep

A CLI tool that scans a CSV, fixes common data quality issues, and writes a plain-English report explaining exactly what it changed and why.

## Setup

Requires Python 3.9+ and pandas.

```bash
git clone <this-repo-url>
cd data-sweep
pip install pandas
```

## Usage

The CSV you want to clean does **not** need to live inside this repo — pass any path, relative or absolute:

```bash
python3 run.py yourfile.csv
python3 run.py /path/to/yourfile.csv
python3 /path/to/data-sweep/run.py yourfile.csv
```

You'll be prompted to pick which columns you're actually testing with (so id/name-style junk columns never get touched). Hit enter to keep all columns.

Output: `cleaned_yourfile.csv` (the fixed data) and `report_yourfile.md` (what happened) — both written to your **current working directory**, not the input file's directory. `cd` into the folder you want the output in before running, if that matters to you.

**Flags:**

```bash
python3 run.py yourfile.csv --columns "age,city,status"   # skip the prompt, pick columns explicitly
python3 run.py yourfile.csv --all-columns                 # skip the prompt, keep everything
python3 run.py yourfile.csv --missing-threshold 0.3        # drop a column if >30% missing (default 50%)
```

## What it detects and fixes

| Issue | Detection | Fix |
|---|---|---|
| Duplicate rows | Exact row copies | Drop, keep first occurrence |
| Constant columns | Only one unique value | Drop (no information) |
| Missing values | Nulls under the missing-threshold | Fill with median (numeric) or mode (text) |
| Missing values (severe) | Nulls over the missing-threshold | Drop column |
| Categorical — ordinal | Text values matching a known ordered scale (e.g. low/medium/high) | Ordinal-encode to preserve order |
| Categorical — one-hot | Low-cardinality text column | One-hot encode |
| Categorical — identifier | Text column that's mostly unique per row (e.g. names) | Drop (not a real category) |
| Categorical — high-cardinality | Too many unique values to one-hot safely | Drop |
| Outliers | Values outside 1.5×IQR from Q1/Q3 | Cap to the IQR bounds |

## Sample report

```markdown
# data-sweep report

## Summary
- Rows: 11 -> 10
- Columns: 5 -> 8
- Issues found: 5

## Findings

### 1. (all) — Duplicate Rows
- **Confidence:** 100%
- **Detail:** Found 1 duplicate row(s) that are exact copies of other rows.
- **Action taken:** Removed 1 duplicate row(s), keeping the first occurrence.

### 2. age — Missing Values
- **Confidence:** 100%
- **Detail:** Column 'age' is missing 3 value(s) (27%).
- **Action taken:** Filled missing values with the column's median (39.0).

### 3. city — Categorical Encoding
- **Confidence:** 90%
- **Detail:** Column 'city' is a text column with 4 unique value(s), which most models can't use directly.
- **Action taken:** One-hot encoded into 4 column(s).

### 4. status — Categorical Encoding
- **Confidence:** 90%
- **Detail:** Column 'status' is a text column with 2 unique value(s), which most models can't use directly.
- **Action taken:** One-hot encoded into 2 column(s).

### 5. region — Constant Column
- **Confidence:** 100%
- **Detail:** Column 'region' has only one unique value ('us') across all rows, so it carries no information.
- **Action taken:** Dropped column.
```
