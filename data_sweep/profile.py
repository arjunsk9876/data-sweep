from typing import Optional

import pandas as pd

from data_sweep.findings import Finding
from data_sweep.ordinal import match_ordinal_scale


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

    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        findings.append(Finding(
            column="(all)",
            issue_type="duplicate_rows",
            confidence=1.0,
            detail=f"Found {dup_count} duplicate row(s) that are exact copies of other rows.",
            action_taken=f"Removed {dup_count} duplicate row(s), keeping the first occurrence.",
        ))

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

        if not pd.api.types.is_numeric_dtype(series):
            non_null_raw = series.dropna()
            if len(non_null_raw) > 0:
                coerced_raw = pd.to_numeric(non_null_raw, errors="coerce")
                parsed_ratio = coerced_raw.notna().mean()
                if mixed_type_threshold <= parsed_ratio < 1.0:
                    stray_values = sorted(set(non_null_raw[coerced_raw.isna()].astype(str)))
                    shown = ", ".join(stray_values[:5]) + ("..." if len(stray_values) > 5 else "")
                    findings.append(Finding(
                        column=col,
                        issue_type="mixed_type_column",
                        confidence=round(parsed_ratio, 2),
                        detail=f"Column '{col}' is {parsed_ratio:.0%} numeric but contains non-numeric placeholder(s): {shown}.",
                        action_taken=f"Treated {shown} as missing and converted column to numeric.",
                    ))
                    series = pd.to_numeric(series, errors="coerce")

        non_null = series.dropna()
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

        missing_count = int(series.isna().sum())
        if missing_count > 0:
            missing_pct = missing_count / len(df)
            if missing_pct > missing_threshold:
                findings.append(Finding(
                    column=col,
                    issue_type="missing_values",
                    confidence=1.0,
                    detail=f"Column '{col}' is missing {missing_pct:.0%} of its values, above the {missing_threshold:.0%} threshold.",
                    action_taken="Dropped column (too much missing data to impute reliably).",
                ))
                continue
            else:
                if pd.api.types.is_numeric_dtype(series):
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

        if pd.api.types.is_numeric_dtype(series):
            q1, q3 = non_null.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_count = int(((non_null < lower) | (non_null > upper)).sum())
                if outlier_count > 0:
                    findings.append(Finding(
                        column=col,
                        issue_type="outliers",
                        confidence=0.7,
                        detail=f"Column '{col}' has {outlier_count} value(s) outside the IQR bounds [{lower:.2f}, {upper:.2f}].",
                        action_taken=f"Capped values to [{lower:.2f}, {upper:.2f}].",
                    ))

        if not pd.api.types.is_numeric_dtype(series):
            scale = match_ordinal_scale(non_null)
            if scale:
                findings.append(Finding(
                    column=col,
                    issue_type="categorical_encoding",
                    confidence=0.95,
                    detail=f"Column '{col}' values follow a natural order ({' < '.join(scale)}).",
                    action_taken=f"Ordinal encoded using order: {' < '.join(scale)}.",
                ))
            elif unique_count / len(df) > max_unique_ratio:
                findings.append(Finding(
                    column=col,
                    issue_type="categorical_encoding",
                    confidence=0.85,
                    detail=f"Column '{col}' has {unique_count} unique value(s) across {len(df)} row(s) ({unique_count / len(df):.0%} unique), which looks like an identifier or free text rather than a category.",
                    action_taken="Dropped column (values are effectively unique per row, not encodable as categories).",
                ))
            elif unique_count <= max_categories:
                findings.append(Finding(
                    column=col,
                    issue_type="categorical_encoding",
                    confidence=0.9,
                    detail=f"Column '{col}' is a text column with {unique_count} unique value(s), which most models can't use directly.",
                    action_taken=f"One-hot encoded into {unique_count} column(s).",
                ))
            elif unique_count <= max_categories_bucketed:
                kept = max_categories - 1
                findings.append(Finding(
                    column=col,
                    issue_type="categorical_encoding",
                    confidence=0.75,
                    detail=f"Column '{col}' has {unique_count} unique values, above the {max_categories} threshold for safe one-hot encoding, but not so many that the long tail is worthless.",
                    action_taken=f"Kept the {kept} most frequent value(s), bucketed the rest into 'other', then one-hot encoded into {max_categories} column(s).",
                ))
            else:
                findings.append(Finding(
                    column=col,
                    issue_type="categorical_encoding",
                    confidence=0.8,
                    detail=f"Column '{col}' has {unique_count} unique values, above the {max_categories_bucketed} threshold even for bucketing rare categories.",
                    action_taken="Dropped column (too many categories to encode usefully).",
                ))

    return findings
