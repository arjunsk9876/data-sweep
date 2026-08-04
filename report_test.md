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
