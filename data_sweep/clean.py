import pandas as pd


def clean(df: pd.DataFrame, missing_threshold: float = 0.5) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index(drop=True)

    for col in list(df.columns):
        unique_count = df[col].nunique(dropna=True)

        if unique_count <= 1:
            df = df.drop(columns=[col])
            continue

        missing_count = df[col].isna().sum()
        if missing_count == 0:
            continue

        missing_pct = missing_count / len(df)
        if missing_pct > missing_threshold:
            df = df.drop(columns=[col])
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].median()
            else:
                fill_value = df[col].mode().iloc[0]
            df[col] = df[col].fillna(fill_value)

    return df
