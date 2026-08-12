import pandas as pd

from data_sweep.detectors.categorical import classify_categorical
from data_sweep.detectors.constant import is_constant
from data_sweep.detectors.missing import missing_fill_decision
from data_sweep.detectors.mixed_type import coerce_mixed_type_column
from data_sweep.detectors.outliers import compute_iqr_bounds
from data_sweep.ordinal import ordinal_mapping


def clean(
    df: pd.DataFrame,
    missing_threshold: float = 0.5,
    max_categories: int = 15,
    max_unique_ratio: float = 0.5,
    mixed_type_threshold: float = 0.8,
    max_categories_bucketed: int = 50,
) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index(drop=True)

    for col in list(df.columns):
        df[col], _ = coerce_mixed_type_column(col, df[col], mixed_type_threshold)

        non_null = df[col].dropna()
        unique_count = non_null.nunique()

        if is_constant(unique_count):
            df = df.drop(columns=[col])
            continue

        if df[col].isna().sum() > 0:
            should_drop, fill_value = missing_fill_decision(df[col], non_null, len(df), missing_threshold)
            if should_drop:
                df = df.drop(columns=[col])
                continue
            df[col] = df[col].fillna(fill_value)

        # re-read fresh (post-fill): a filled value is now part of the column,
        # so bounds/bucketing must be computed from what's actually there, not
        # from the pre-fill snapshot. unique_count deliberately stays stale,
        # matching the original clean()'s behavior.
        non_null = df[col].dropna()

        if pd.api.types.is_numeric_dtype(df[col]):
            bounds = compute_iqr_bounds(non_null)
            if bounds:
                lower, upper = bounds
                df[col] = df[col].clip(lower, upper)

        if not pd.api.types.is_numeric_dtype(df[col]):
            decision = classify_categorical(non_null, unique_count, len(df), max_categories, max_unique_ratio, max_categories_bucketed)

            if decision.tier == "ordinal":
                df[col] = df[col].map(ordinal_mapping(df[col], decision.scale))
            elif decision.tier == "identifier":
                df = df.drop(columns=[col])
            elif decision.tier == "one_hot":
                dummies = pd.get_dummies(df[col], prefix=col)
                col_pos = df.columns.get_loc(col)
                df = pd.concat([df.iloc[:, :col_pos], dummies, df.iloc[:, col_pos + 1:]], axis=1)
            elif decision.tier == "bucketed":
                bucketed = df[col].where(df[col].isin(decision.kept_categories), other="other")
                dummies = pd.get_dummies(bucketed, prefix=col)
                col_pos = df.columns.get_loc(col)
                df = pd.concat([df.iloc[:, :col_pos], dummies, df.iloc[:, col_pos + 1:]], axis=1)
            else:  # drop
                df = df.drop(columns=[col])

    return df
