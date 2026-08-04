import pandas as pd

from data_sweep.ordinal import match_ordinal_scale, ordinal_mapping


def clean(df: pd.DataFrame, missing_threshold: float = 0.5, max_categories: int = 15) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index(drop=True)

    for col in list(df.columns):
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

        if not pd.api.types.is_numeric_dtype(df[col]):
            scale = match_ordinal_scale(df[col].dropna())
            if scale:
                df[col] = df[col].map(ordinal_mapping(df[col], scale))
            elif unique_count <= max_categories:
                dummies = pd.get_dummies(df[col], prefix=col)
                col_pos = df.columns.get_loc(col)
                df = pd.concat([df.iloc[:, :col_pos], dummies, df.iloc[:, col_pos + 1:]], axis=1)
            else:
                df = df.drop(columns=[col])

    return df
