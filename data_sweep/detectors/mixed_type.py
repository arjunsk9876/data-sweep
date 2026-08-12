from typing import Optional

import pandas as pd

from data_sweep.findings import Finding


def coerce_mixed_type_column(col: str, series: pd.Series, mixed_type_threshold: float) -> tuple[pd.Series, Optional[Finding]]:
    if pd.api.types.is_numeric_dtype(series):
        return series, None

    non_null_raw = series.dropna()
    if len(non_null_raw) == 0:
        return series, None

    coerced_raw = pd.to_numeric(non_null_raw, errors="coerce")
    parsed_ratio = coerced_raw.notna().mean()
    if not (mixed_type_threshold <= parsed_ratio < 1.0):
        return series, None

    stray_values = sorted(set(non_null_raw[coerced_raw.isna()].astype(str)))
    shown = ", ".join(stray_values[:5]) + ("..." if len(stray_values) > 5 else "")
    finding = Finding(
        column=col,
        issue_type="mixed_type_column",
        confidence=round(parsed_ratio, 2),
        detail=f"Column '{col}' is {parsed_ratio:.0%} numeric but contains non-numeric placeholder(s): {shown}.",
        action_taken=f"Treated {shown} as missing and converted column to numeric.",
    )
    return pd.to_numeric(series, errors="coerce"), finding
