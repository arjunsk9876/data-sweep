import pandas as pd

from data_sweep.findings import Finding


def profile(df: pd.DataFrame, missing_threshold: float = 0.5) -> list[Finding]:
    findings = []

    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        findings.append(Finding(
            column="(all)",
            issue_type="duplicate_rows",
            confidence=1.0,
            detail=f"Found {dup_count} duplicate row(s) that are exact copies of other rows.",
            action_taken=f"Removed {dup_count} duplicate row(s), keeping the first occurrence.",
        ))

    for col in df.columns:
        non_null = df[col].dropna()
        unique_count = non_null.nunique()

        if unique_count <= 1:
            if unique_count == 1:
                detail = f"Column '{col}' has only one unique value ('{non_null.iloc[0]}') across all rows, so it carries no information."
            else:
                detail = f"Column '{col}' has no non-missing values, so it carries no information."
            findings.append(Finding(
                column=col,
                issue_type="constant_column",
                confidence=1.0,
                detail=detail,
                action_taken="Dropped column.",
            ))
            continue

        missing_count = int(df[col].isna().sum())
        if missing_count == 0:
            continue

        missing_pct = missing_count / len(df)
        if missing_pct > missing_threshold:
            findings.append(Finding(
                column=col,
                issue_type="missing_values",
                confidence=1.0,
                detail=f"Column '{col}' is missing {missing_pct:.0%} of its values, above the {missing_threshold:.0%} threshold.",
                action_taken="Dropped column (too much missing data to impute reliably).",
            ))
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = non_null.median()
                strategy = "median"
            else:
                fill_value = non_null.mode().iloc[0]
                strategy = "mode"
            findings.append(Finding(
                column=col,
                issue_type="missing_values",
                confidence=1.0,
                detail=f"Column '{col}' is missing {missing_count} value(s) ({missing_pct:.0%}).",
                action_taken=f"Filled missing values with the column's {strategy} ({fill_value}).",
            ))

    return findings
