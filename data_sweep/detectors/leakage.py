from typing import Optional

import pandas as pd

from data_sweep.findings import Finding


def find_data_leakage(df: pd.DataFrame, target: Optional[str], leakage_threshold: float) -> list[Finding]:
    if not (target and target in df.columns and pd.api.types.is_numeric_dtype(df[target])):
        return []

    findings = []
    for col in df.columns:
        if col == target or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        corr = df[col].corr(df[target])
        if pd.notna(corr) and abs(corr) > leakage_threshold:
            findings.append(Finding(
                column=col,
                issue_type="data_leakage",
                confidence=round(abs(corr), 2),
                detail=f"Column '{col}' is {abs(corr):.0%} correlated with target '{target}' — too close to be a real predictor, likely leaks the label.",
                action_taken="Flagged only (not auto-dropped — review before training on this column).",
            ))
    return findings
