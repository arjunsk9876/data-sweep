# data-sweep report

## Summary
- Rows: 11 -> 11
- Columns: 7 -> 12
- Issues found: 6

## Findings

### 1. name — Categorical Encoding
- **Confidence:** 85%
- **Detail:** Column 'name' has 10 unique value(s) across 11 row(s) (91% unique), which looks like an identifier or free text rather than a category.
- **Action taken:** Dropped column (values are effectively unique per row, not encodable as categories).

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

### 6.  color — Categorical Encoding
- **Confidence:** 90%
- **Detail:** Column ' color' is a text column with 4 unique value(s), which most models can't use directly.
- **Action taken:** One-hot encoded into 4 column(s).
