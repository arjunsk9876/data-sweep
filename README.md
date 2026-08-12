# data-sweep

A CLI tool that scans a CSV, fixes common data quality issues, and writes a plain-English report explaining exactly what it changed and why.

## Setup

Requires Python 3.9+.

**Recommended — install as a command:**

```bash
git clone <this-repo-url>
cd data-sweep
pip install -e .
```

This gives you a `data-sweep` command on your PATH (see Usage below).

**No packaging, just the script** — if you'd rather not install anything beyond the one dependency:

```bash
git clone <this-repo-url>
cd data-sweep
pip install pandas
```

Then run it directly with `python3 run.py yourfile.csv` instead of `data-sweep yourfile.csv` — every example below works either way, just swap the command.

## Usage

The CSV you want to clean does **not** need to live inside this repo — pass any path, relative or absolute:

```bash
data-sweep yourfile.csv
data-sweep /path/to/yourfile.csv
```

(Using the script directly instead: `python3 run.py yourfile.csv`, or `python3 /path/to/data-sweep/run.py yourfile.csv` from elsewhere.)

You'll be prompted to pick which columns you're actually testing with (so id/name-style junk columns never get touched). Hit enter to keep all columns.

Output: `cleaned_yourfile.csv` (the fixed data) and `report_yourfile.md` (what happened) — both written to your **current working directory**, not the input file's directory. `cd` into the folder you want the output in before running, if that matters to you.

**Flags:**

```bash
data-sweep yourfile.csv --columns "age,city,status"   # skip the prompt, pick columns explicitly
data-sweep yourfile.csv --all-columns                 # skip the prompt, keep everything
data-sweep yourfile.csv --missing-threshold 0.3        # drop a column if >30% missing (default 50%)
data-sweep yourfile.csv --target label                 # flag columns leaking the target column, plus class imbalance
data-sweep yourfile.csv --dry-run                       # preview the report only, don't write a cleaned CSV
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
| Categorical — high-cardinality | Too many unique values to one-hot safely (even after bucketing) | Drop |
| Categorical — bucketed | Too many uniques for direct one-hot, but not hopeless | Keep the most frequent values, merge the rest into `other`, then one-hot |
| Outliers | Values outside 1.5×IQR from Q1/Q3 | Cap to the IQR bounds |
| Mixed-type column | Object column that's mostly numeric but has stray non-numeric placeholders (e.g. `unknown`) | Coerce strays to missing, convert column to numeric, then fill like any other missing value |
| Data leakage | Numeric column >95% correlated with `--target` | Flag only (no auto-fix — you decide whether to drop it) |
| Multicollinearity | Two numeric columns >95% correlated with each other | Flag only (consider dropping one) |
| Class imbalance | `--target` has a majority class over 90% of rows | Flag only (consider resampling, class weights, or a non-accuracy metric) |

## Sample report

Basic run, no `--target`:

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

Run with `--target label` on a messier dataset (mixed-type column, two redundant numeric columns, a rare-heavy category, an imbalanced label):

```markdown
# data-sweep report

## Summary
- Rows: 300 -> 300
- Columns: 5 -> 20
- Issues found: 6

## Findings

### 1. label — Class Imbalance
- **Confidence:** 98%
- **Detail:** Target 'label' is 98% class 'no' — severe class imbalance, accuracy will be a misleading metric.
- **Action taken:** Flagged only (consider resampling, class weights, or a metric like F1/AUC instead of accuracy).

### 2. income & income2 — Multicollinearity
- **Confidence:** 100%
- **Detail:** Columns 'income' and 'income2' are 100% correlated with each other — redundant information, can destabilize model coefficients.
- **Action taken:** Flagged only (consider dropping one of the two).

### 3. age — Mixed Type Column
- **Confidence:** 97%
- **Detail:** Column 'age' is 97% numeric but contains non-numeric placeholder(s): unknown.
- **Action taken:** Treated unknown as missing and converted column to numeric.

### 4. age — Missing Values
- **Confidence:** 100%
- **Detail:** Column 'age' is missing 20 value(s) (7%).
- **Action taken:** Filled missing values with the column's median (43.0).

### 5. city — Categorical Encoding
- **Confidence:** 75%
- **Detail:** Column 'city' has 25 unique values, above the 15 threshold for safe one-hot encoding, but not so many that the long tail is worthless.
- **Action taken:** Kept the 14 most frequent value(s), bucketed the rest into 'other', then one-hot encoded into 15 column(s).

### 6. label — Categorical Encoding
- **Confidence:** 90%
- **Detail:** Column 'label' is a text column with 2 unique value(s), which most models can't use directly.
- **Action taken:** One-hot encoded into 2 column(s).
```

With `--dry-run`, you get `report_yourfile.md` only — no `cleaned_yourfile.csv` is written, and the terminal says so explicitly.

## Development

Detection logic lives in `data_sweep/detectors/` — one module per concern (duplicates, constant, missing, outliers, mixed-type, categorical, leakage, imbalance, multicollinearity). `data_sweep/profile.py` is a thin orchestrator that calls each detector and reports what it finds; `data_sweep/clean.py` calls the same detector decision logic to actually apply the fix, so the two never drift apart.

Install with dev dependencies (pytest, mypy):

```bash
pip install -e ".[dev]"
```

Run the test suite:

```bash
python -m pytest
```

Run type checks:

```bash
mypy data_sweep run.py
```

`tests/fixtures/sample.csv` is a small, deterministic dataset that exercises every detector branch (duplicate row, constant column, all-null column, missing under/over threshold, IQR outliers, ordinal/one-hot/bucketed/identifier/dropped categorical tiers, mixed-type column, multicollinear pair, leaky feature, imbalanced target) — useful both as a test fixture and as a quick manual smoke test:

```bash
data-sweep tests/fixtures/sample.csv --all-columns --target target
```

CI (`.github/workflows/ci.yml`) runs both pytest and mypy on every push and pull request.
