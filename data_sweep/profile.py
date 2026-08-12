from typing import Optional

import pandas as pd

from data_sweep.detectors.categorical import find_categorical_encoding
from data_sweep.detectors.constant import find_constant_column
from data_sweep.detectors.duplicates import find_duplicate_rows
from data_sweep.detectors.imbalance import find_class_imbalance
from data_sweep.detectors.leakage import find_data_leakage
from data_sweep.detectors.missing import find_missing_values
from data_sweep.detectors.mixed_type import coerce_mixed_type_column
from data_sweep.detectors.multicollinearity import find_multicollinearity
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

    # clean() drops duplicate rows before doing anything else, so every other
    # detector needs to look at the same post-dedup data clean() will actually
    # act on — otherwise a stat like a missing-value percentage, an IQR bound,
    # or a unique-value ratio can land on a different side of a threshold here
    # than it does in clean(), and the report ends up describing an action
    # clean() didn't actually take.
    df = df.drop_duplicates()

    findings.extend(find_data_leakage(df, target, leakage_threshold))
    findings.extend(find_class_imbalance(df, target, max_categories, imbalance_threshold))
    findings.extend(find_multicollinearity(df, target, multicollinearity_threshold))

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
