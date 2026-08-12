from typing import Optional

import pandas as pd

from data_sweep.findings import Finding


def find_constant_column(col: str, non_null: pd.Series, unique_count: int) -> Optional[Finding]:
    if unique_count > 1:
        return None
    if unique_count == 1:
        detail = f"Column '{col}' has only one unique value ('{non_null.iloc[0]}') across all rows, so it carries no information."
    else:
        detail = f"Column '{col}' has no non-missing values, so it carries no information."
    return Finding(
        column=col,
        issue_type="constant_column",
        confidence=1.0,
        detail=detail,
        action_taken="Dropped column.",
    )
