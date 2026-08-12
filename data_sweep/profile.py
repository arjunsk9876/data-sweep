from typing import Optional

import pandas as pd

from data_sweep.detectors.categorical import find_categorical_encoding
from data_sweep.detectors.constant import find_constant_column
from data_sweep.detectors.duplicates import find_duplicate_rows
from data_sweep.detectors.mixed_type import coerce_mixed_type_column
from data_sweep.detectors.missing import find_missing_values
from data_sweep.detectors.outliers import find_outliers
from data_sweep.findings import Finding


def profile(
    df: pd.DataFrame,
    missing_threshold: float = 0.5,
    max_categories: int = 15,
    max_unique_ratio: float = 0.5,
    target: Optional[str] = None,
    leakage_threshold: float = 0.95,
    mixed_type_threshold: float = 0.8,
    multicollinearity_threshold: float = 0.95,
    imbalance_threshold: float = 0.9,
    max_categories_bucketed: int = 50,
) -> list[Finding]:
    findings = []

    findings.extend(find_duplicate_rows(df))

    if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
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

    if target and target in df.columns:
        target_non_null = df[target].dropna()
        if len(target_non_null) > 0 and target_non_null.nunique() <= max_categories:
            value_counts = target_non_null.value_counts(normalize=True)
            top_prop = value_counts.iloc[0]
            if top_prop > imbalance_threshold:
                top_class = value_counts.index[0]
                findings.append(Finding(
                    column=target,
                    issue_type="class_imbalance",
                    confidence=round(top_prop, 2),
                    detail=f"Target '{target}' is {top_prop:.0%} class '{top_class}' — severe class imbalance, accuracy will be a misleading metric.",
                    action_taken="Flagged only (consider resampling, class weights, or a metric like F1/AUC instead of accuracy).",
                ))

    numeric_cols = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
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

    for col in df.columns:
        series = df[col]

        series, mixed_type_finding = coerce_mixed_type_column(col, series, mixed_type_threshold)
        if mixed_type_finding:
            findings.append(mixed_type_finding)

        non_null = series.dropna()
        unique_count = non_null.nunique()

        constant_finding = find_constant_column(col, non_null, unique_count)
        if constant_finding:
            findings.append(constant_finding)
            continue

        missing_count = int(series.isna().sum())
        missing_finding = find_missing_values(col, series, non_null, len(df), missing_threshold)
        if missing_finding:
            findings.append(missing_finding)
            if missing_count / len(df) > missing_threshold:
                continue

        outlier_finding = find_outliers(col, series, non_null)
        if outlier_finding:
            findings.append(outlier_finding)

        categorical_finding = find_categorical_encoding(
            col, series, non_null, unique_count, len(df),
            max_categories, max_unique_ratio, max_categories_bucketed,
        )
        if categorical_finding:
            findings.append(categorical_finding)

    return findings
