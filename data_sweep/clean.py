import pandas as pd

from data_sweep.ordinal import match_ordinal_scale, ordinal_mapping


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
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_null_raw = df[col].dropna()
            if len(non_null_raw) > 0:
                coerced_raw = pd.to_numeric(non_null_raw, errors="coerce")
                parsed_ratio = coerced_raw.notna().mean()
                if mixed_type_threshold <= parsed_ratio < 1.0:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        unique_count = df[col].nunique(dropna=True)

        if unique_count <= 1:
            df = df.drop(columns=[col])
            continue

        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_pct = missing_count / len(df)
            if missing_pct > missing_threshold:
                df = df.drop(columns=[col])
                continue
            else:
                if pd.api.types.is_numeric_dtype(df[col]):
                    fill_value = df[col].median()
                else:
                    fill_value = df[col].mode().iloc[0]
                df[col] = df[col].fillna(fill_value)

        if pd.api.types.is_numeric_dtype(df[col]):
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                df[col] = df[col].clip(lower, upper)

        if not pd.api.types.is_numeric_dtype(df[col]):
            scale = match_ordinal_scale(df[col].dropna())
            if scale:
                df[col] = df[col].map(ordinal_mapping(df[col], scale))
            elif unique_count / len(df) > max_unique_ratio:
                df = df.drop(columns=[col])
            elif unique_count <= max_categories:
                dummies = pd.get_dummies(df[col], prefix=col)
                col_pos = df.columns.get_loc(col)
                df = pd.concat([df.iloc[:, :col_pos], dummies, df.iloc[:, col_pos + 1:]], axis=1)
            elif unique_count <= max_categories_bucketed:
                top_categories = df[col].value_counts().nlargest(max_categories - 1).index
                bucketed = df[col].where(df[col].isin(top_categories), other="other")
                dummies = pd.get_dummies(bucketed, prefix=col)
                col_pos = df.columns.get_loc(col)
                df = pd.concat([df.iloc[:, :col_pos], dummies, df.iloc[:, col_pos + 1:]], axis=1)
            else:
                df = df.drop(columns=[col])

    return df
