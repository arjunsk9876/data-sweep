from typing import Optional

import pandas as pd

from data_sweep.findings import Finding
from data_sweep.ordinal import match_ordinal_scale


def find_categorical_encoding(
    col: str,
    series: pd.Series,
    non_null: pd.Series,
    unique_count: int,
    total_rows: int,
    max_categories: int,
    max_unique_ratio: float,
    max_categories_bucketed: int,
) -> Optional[Finding]:
    if pd.api.types.is_numeric_dtype(series):
        return None

    scale = match_ordinal_scale(non_null)
    if scale:
        return Finding(
            column=col,
            issue_type="categorical_encoding",
            confidence=0.95,
            detail=f"Column '{col}' values follow a natural order ({' < '.join(scale)}).",
            action_taken=f"Ordinal encoded using order: {' < '.join(scale)}.",
        )

    if unique_count / total_rows > max_unique_ratio:
        return Finding(
            column=col,
            issue_type="categorical_encoding",
            confidence=0.85,
            detail=f"Column '{col}' has {unique_count} unique value(s) across {total_rows} row(s) ({unique_count / total_rows:.0%} unique), which looks like an identifier or free text rather than a category.",
            action_taken="Dropped column (values are effectively unique per row, not encodable as categories).",
        )

    if unique_count <= max_categories:
        return Finding(
            column=col,
            issue_type="categorical_encoding",
            confidence=0.9,
            detail=f"Column '{col}' is a text column with {unique_count} unique value(s), which most models can't use directly.",
            action_taken=f"One-hot encoded into {unique_count} column(s).",
        )

    if unique_count <= max_categories_bucketed:
        kept = max_categories - 1
        return Finding(
            column=col,
            issue_type="categorical_encoding",
            confidence=0.75,
            detail=f"Column '{col}' has {unique_count} unique values, above the {max_categories} threshold for safe one-hot encoding, but not so many that the long tail is worthless.",
            action_taken=f"Kept the {kept} most frequent value(s), bucketed the rest into 'other', then one-hot encoded into {max_categories} column(s).",
        )

    return Finding(
        column=col,
        issue_type="categorical_encoding",
        confidence=0.8,
        detail=f"Column '{col}' has {unique_count} unique values, above the {max_categories_bucketed} threshold even for bucketing rare categories.",
        action_taken="Dropped column (too many categories to encode usefully).",
    )
