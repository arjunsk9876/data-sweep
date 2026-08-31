# data-sweep

A CLI tool that scans a CSV, fixes common data quality issues, audits train/test splits for hidden entity leakage, and writes a plain-English report explaining exactly what it found and why.

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

Then run it directly with `python3 run.py clean yourfile.csv` instead of `data-sweep clean yourfile.csv` — every example below works either way, just swap the command.

## Usage

data-sweep has two subcommands: `clean` (profile + fix data quality issues) and `audit` (find hidden entity/group leakage between a train and test split — see [Entity leakage detection](#entity-leakage-detection) below). If you used data-sweep before it had subcommands, `data-sweep yourfile.csv` is now `data-sweep clean yourfile.csv`.

The CSV you want to clean does **not** need to live inside this repo — pass any path, relative or absolute:

```bash
data-sweep clean yourfile.csv
data-sweep clean /path/to/yourfile.csv
```

(Using the script directly instead: `python3 run.py clean yourfile.csv`, or `python3 /path/to/data-sweep/run.py clean yourfile.csv` from elsewhere.)

You'll be prompted to pick which columns you're actually testing with (so id/name-style junk columns never get touched). Hit enter to keep all columns.

Output: `cleaned_yourfile.csv` (the fixed data) and `report_yourfile.md` (what happened) — both written to your **current working directory**, not the input file's directory. `cd` into the folder you want the output in before running, if that matters to you.

**Flags:**

```bash
data-sweep clean yourfile.csv --columns "age,city,status"   # skip the prompt, pick columns explicitly
data-sweep clean yourfile.csv --all-columns                 # skip the prompt, keep everything
data-sweep clean yourfile.csv --missing-threshold 0.3        # drop a column if >30% missing (default 50%)
data-sweep clean yourfile.csv --target label                 # flag columns leaking the target column, plus class imbalance
data-sweep clean yourfile.csv --dry-run                       # preview the report only, don't write a cleaned CSV
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

## Entity leakage detection

The flagship feature of `audit`: automatically finds columns that act as an entity or group key — a customer id, household id, device id, anything where multiple rows belong to the same real-world entity — and checks whether that same entity shows up in *both* your train and test files. If it does, your test score is inflated: the model isn't actually being evaluated on unseen entities.

No configuration needed. You never have to tell it which column is the entity key — it's inferred straight from the data: how many distinct values a column has relative to the row count, whether the values look ID-shaped (UUID, hash, zero-padded number, alphanumeric code), and whether the column name hints at it (`id`, `key`, `uuid`, and similar). None of those signals require you to already know the answer.

```bash
data-sweep audit train.csv --test test.csv
```

**Before** — say `train.csv` and `test.csv` were split randomly by row instead of by customer, so the same customer's other purchases end up on both sides:

```
Found 1 possible leak between train and test:

Column 'customer_id' looks like an entity/group key, and 42 of 210 test values
(20.0%) also appear in the training data.
  Example overlapping values: C1042, C1058, C1091
  This means rows for the same entity can land in both train and test, so a
  model can partly memorize the entity instead of learning to generalize --
  test performance can look better than it will be on genuinely unseen
  entities.
  Recommendation: split train/test by 'customer_id' (a group/entity split)
  instead of by row, so each entity appears in only one side of the split.
```

### Turning a finding into a fix

Finding the leak is only half the problem — you'd still have to know that `GroupShuffleSplit` (not `train_test_split`) is the fix, and write the code yourself. Add `--fix` to get a runnable snippet instead, using the actual column data-sweep found:

```bash
data-sweep audit train.csv --test test.csv --fix
```

```
Found 1 possible leak between train and test:

Column 'customer_id' looks like an entity/group key, and 42 of 210 test values (20.0%) also appear in the training data.
  Example overlapping values: C1000, C1001, C1002
  This means rows for the same entity can land in both train and test, so a model can partly memorize the entity instead of learning to generalize -- test performance can look better than it will be on genuinely unseen entities.
  Recommendation: split train/test by 'customer_id' (a group/entity split) instead of by row, so each entity appears in only one side of the split.

from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(df, groups=df['customer_id']))

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]
```

That's not a template with a blank to fill in — `'customer_id'` is the real column name data-sweep detected, ready to paste into your own script (`df` there is your combined dataframe: `pd.concat([train_df, test_df])`, since the whole point is re-splitting cleanly from scratch rather than patching the existing, already-contaminated split). Run `data-sweep audit train.csv --test test.csv --fix-file fix.py` instead to write it straight to a file rather than print it.

**After** — re-split by `customer_id` (e.g. with the generated fix above) and re-run:

```
No entity/group leakage detected between train and test.
```

**Single-file mode** — run it on just one file to see what data-sweep considers a candidate entity/group key, even before you've made a split. `--fix` still works here — the risk is about the *future* split/CV you haven't made yet, so the generated code uses `GroupKFold` instead:

```bash
data-sweep audit train.csv --fix
```

```
Detected 1 candidate entity/group key column:
  'customer_id' -- uniqueness ratio 0.333 (signals: uniqueness, format, name)

No --test file was provided, so this was informational only -- no leakage check was run. Pass --test <file> to check these columns for overlap between train and test.

from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(df, groups=df['customer_id']):
    train_fold = df.iloc[train_idx]
    val_fold = df.iloc[val_idx]
    # ... fit/evaluate per fold
```

If more than one column looks like an entity key, the highest-severity one gets fixed and every other candidate gets a one-line comment above the code noting it as worth considering too — nothing gets silently dropped.

`--fix`/`--fix-file` need scikit-learn (`pip install scikit-learn`, or `pip install data-sweep[fix]`) — auditing itself never requires it, only generating a fix that calls into it does. Without it installed, `--fix` prints a plain error after the normal audit report instead of crashing.

## Development

**`clean`**: detection logic lives in `data_sweep/detectors/` — one module per concern (duplicates, constant, missing, outliers, mixed-type, categorical, leakage, imbalance, multicollinearity). `data_sweep/profile.py` is a thin orchestrator that calls each detector and reports what it finds; `data_sweep/clean.py` calls the same detector decision logic to actually apply the fix, so the two never drift apart.

**`audit`**: lives in `data_sweep/entity_leakage/` — `io.py` loads the CSV(s) with clean error handling, `keys.py` scores every column as a candidate entity/group key (uniqueness ratio gates candidacy; ID-format and name-keyword signals only boost ranking among already-qualified candidates, never gate on their own), `leakage.py` checks candidate columns for cross-split value overlap and ranks findings worst-first, and `report.py` turns the results into the plain-language output shown above. `findings.py` defines `EntityLeakageFinding`, the mode-aware shape (`two_file` vs `single_file`) that both the plain-language report and fix generation build on. `fixes.py` renders a runnable fix (`GroupShuffleSplit` or `GroupKFold`, picked by mode) from those findings, with a clean error if scikit-learn isn't installed. `cli.py` wires it all together, including `--fix`/`--fix-file`. `tests/synthetic.py` generates train/test pairs with known, guaranteed ground truth (a real injected leak, a genuinely disjoint split, or no entity structure at all) so detection logic can be tested against cases with a known right answer, not just hand-picked examples; `tests/test_closed_loop.py` goes a step further and proves a generated fix actually resolves the leak it targets, by applying it and re-running the audit.

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
data-sweep clean tests/fixtures/sample.csv --all-columns --target target
```

CI (`.github/workflows/ci.yml`) runs both pytest and mypy on every push and pull request.
