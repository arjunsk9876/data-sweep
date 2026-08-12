from typing import Optional

import pandas as pd

from data_sweep.findings import Finding


def find_multicollinearity(df: pd.DataFrame, target: Optional[str], multicollinearity_threshold: float) -> list[Finding]:
    numeric_cols = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]

    findings = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            a, b = numeric_cols[i], numeric_cols[j]
            corr = df[a].corr(df[b])
            if pd.notna(corr) and abs(corr) > multicollinearity_threshold:
                findings.append(Finding(
                    column=f"{a} & {b}",
                    issue_type="multicollinearity",
                    confidence=round(abs(corr), 2),
                    detail=f"Columns '{a}' and '{b}' are {abs(corr):.0%} correlated with each other — redundant information, can destabilize model coefficients.",
                    action_taken="Flagged only (consider dropping one of the two).",
                ))
    return findings
