from typing import Optional

import pandas as pd

from data_sweep.findings import Finding


def missing_fill_decision(series: pd.Series, non_null: pd.Series, total_rows: int, missing_threshold: float) -> tuple[bool, object]:
    """Returns (should_drop, fill_value). fill_value is None when should_drop is True."""
    missing_pct = int(series.isna().sum()) / total_rows
    if missing_pct > missing_threshold:
        return True, None
    if pd.api.types.is_numeric_dtype(series):
        return False, non_null.median()
    return False, non_null.mode().iloc[0]


def find_missing_values(col: str, series: pd.Series, non_null: pd.Series, total_rows: int, missing_threshold: float) -> Optional[Finding]:
    missing_count = int(series.isna().sum())
    if missing_count == 0:
        return None

    missing_pct = missing_count / total_rows
    should_drop, fill_value = missing_fill_decision(series, non_null, total_rows, missing_threshold)

    if should_drop:
        return Finding(
            column=col,
            issue_type="missing_values",
            confidence=1.0,
            detail=f"Column '{col}' is missing {missing_pct:.0%} of its values, above the {missing_threshold:.0%} threshold.",
            action_taken="Dropped column (too much missing data to impute reliably).",
        )

    strategy = "median" if pd.api.types.is_numeric_dtype(series) else "mode"
    return Finding(
        column=col,
        issue_type="missing_values",
        confidence=1.0,
        detail=f"Column '{col}' is missing {missing_count} value(s) ({missing_pct:.0%}).",
        action_taken=f"Filled missing values with the column's {strategy} ({fill_value}).",
    )
