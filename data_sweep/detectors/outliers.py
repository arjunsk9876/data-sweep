from typing import Optional

import pandas as pd

from data_sweep.findings import Finding


def find_outliers(col: str, series: pd.Series, non_null: pd.Series) -> Optional[Finding]:
    if not pd.api.types.is_numeric_dtype(series):
        return None

    q1, q3 = non_null.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr <= 0:
        return None

    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_count = int(((non_null < lower) | (non_null > upper)).sum())
    if outlier_count == 0:
        return None

    return Finding(
        column=col,
        issue_type="outliers",
        confidence=0.7,
        detail=f"Column '{col}' has {outlier_count} value(s) outside the IQR bounds [{lower:.2f}, {upper:.2f}].",
        action_taken=f"Capped values to [{lower:.2f}, {upper:.2f}].",
    )
