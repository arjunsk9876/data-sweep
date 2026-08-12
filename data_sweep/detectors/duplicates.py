import pandas as pd

from data_sweep.findings import Finding


def find_duplicate_rows(df: pd.DataFrame) -> list[Finding]:
    dup_count = int(df.duplicated().sum())
    if dup_count == 0:
        return []
    return [Finding(
        column="(all)",
        issue_type="duplicate_rows",
        confidence=1.0,
        detail=f"Found {dup_count} duplicate row(s) that are exact copies of other rows.",
        action_taken=f"Removed {dup_count} duplicate row(s), keeping the first occurrence.",
    )]
