from typing import Optional

import pandas as pd

from data_sweep.findings import Finding


def find_class_imbalance(df: pd.DataFrame, target: Optional[str], max_categories: int, imbalance_threshold: float) -> list[Finding]:
    if not (target and target in df.columns):
        return []

    target_non_null = df[target].dropna()
    if len(target_non_null) == 0 or target_non_null.nunique() > max_categories:
        return []

    value_counts = target_non_null.value_counts(normalize=True)
    top_prop = value_counts.iloc[0]
    if top_prop <= imbalance_threshold:
        return []

    top_class = value_counts.index[0]
    return [Finding(
        column=target,
        issue_type="class_imbalance",
        confidence=round(top_prop, 2),
        detail=f"Target '{target}' is {top_prop:.0%} class '{top_class}' — severe class imbalance, accuracy will be a misleading metric.",
        action_taken="Flagged only (consider resampling, class weights, or a metric like F1/AUC instead of accuracy).",
    )]
