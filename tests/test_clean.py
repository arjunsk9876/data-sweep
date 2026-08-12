import pandas as pd

from data_sweep.clean import clean


def test_empty_dataframe_stays_empty():
    df = pd.DataFrame()
    result = clean(df)
    assert result.empty


def test_single_row_drops_all_columns():
    df = pd.DataFrame({"a": [1], "b": ["x"]})
    result = clean(df)
    assert result.shape == (1, 0)


def test_all_null_column_dropped():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
    result = clean(df)
    assert list(result.columns) == ["a"]
    assert result["a"].tolist() == [1, 2, 3]


def test_constant_column_dropped():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "region": ["us"] * 4})
    result = clean(df)
    assert list(result.columns) == ["a"]


def test_duplicate_rows_removed():
    df = pd.DataFrame({"a": [1, 2, 1, 3]})
    result = clean(df)
    assert len(result) == 3
    assert sorted(result["a"].tolist()) == [1, 2, 3]
    assert result.duplicated().sum() == 0


def test_missing_under_threshold_filled_with_median():
    df = pd.DataFrame({"a": [10.0, 20.0, 30.0, None]})
    result = clean(df, missing_threshold=0.5)
    assert result["a"].isna().sum() == 0
    assert result["a"].iloc[3] == 20.0


def test_missing_under_threshold_filled_with_mode_for_text():
    # anchor column keeps the two "x" rows from being collapsed as duplicates;
    # low cardinality after fill also triggers one-hot encoding, so the mode
    # fill shows up as an extra True in the majority class's dummy column.
    df = pd.DataFrame({"anchor": [1, 2, 3, 4], "a": ["x", "x", "y", None]})
    result = clean(df, missing_threshold=0.5)
    assert "a" not in result.columns
    assert result["a_x"].tolist() == [True, True, False, True]
    assert result["a_y"].tolist() == [False, False, True, False]


def test_missing_over_threshold_column_dropped():
    # anchor column keeps the three all-null rows from collapsing into one
    # via drop_duplicates (NaN == NaN for dedup purposes); "a" itself needs
    # 2+ distinct non-null values so the constant-column check doesn't catch
    # it before the missing-value check gets a say.
    df = pd.DataFrame({"anchor": [1, 2, 3, 4, 5], "a": [1.0, 2.0, None, None, None]})
    result = clean(df, missing_threshold=0.5)
    assert "a" not in result.columns


def test_outliers_capped_to_iqr_bounds():
    df = pd.DataFrame({
        "anchor": range(11),
        "a": [50, 52, 49, 51, 53, 48, 52, 50, 51, 500, -100],
    })
    result = clean(df)
    assert result["a"].tolist() == [50.0, 52.0, 49.0, 51.0, 53.0, 48.0, 52.0, 50.0, 51.0, 55.75, 45.75]


def test_no_capping_when_values_are_tight():
    df = pd.DataFrame({
        "anchor": range(10),
        "a": [50, 52, 49, 51, 53, 48, 52, 50, 51, 54],
    })
    result = clean(df)
    assert result["a"].tolist() == [50, 52, 49, 51, 53, 48, 52, 50, 51, 54]


def test_mixed_type_column_coerced_to_numeric_above_threshold():
    # the "unknown" stray becomes missing, then gets median-filled like any
    # other numeric missing value.
    df = pd.DataFrame({"a": ["10", "11", "12", "13", "14", "15", "16", "17", "18", "unknown"]})
    result = clean(df)
    assert result["a"].dtype.kind == "f"
    assert result["a"].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 14.0]


def test_mixed_type_column_left_as_text_below_threshold():
    df = pd.DataFrame({"a": ["1", "2", "3", "4", "5", "a", "b", "c", "d", "e"]})
    result = clean(df)
    # dropped as a high-cardinality identifier, not coerced to numeric
    assert "a" not in result.columns


def test_categorical_ordinal_tier_mapped_to_ints():
    df = pd.DataFrame({"anchor": range(9), "a": ["low", "medium", "high"] * 3})
    result = clean(df)
    assert result["a"].tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2]


def test_categorical_one_hot_tier_produces_dummy_columns():
    df = pd.DataFrame({"anchor": range(8), "a": ["red", "green", "blue", "yellow"] * 2})
    result = clean(df)
    assert "a" not in result.columns
    assert sorted(c for c in result.columns if c.startswith("a_")) == ["a_blue", "a_green", "a_red", "a_yellow"]


def test_categorical_identifier_tier_dropped():
    df = pd.DataFrame({"anchor": range(10), "a": [f"name_{i}" for i in range(10)]})
    result = clean(df)
    assert "a" not in result.columns


def test_categorical_bucketed_tier_keeps_top_14_plus_other():
    common_counts = [3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2]  # 14 labels, sums to 34
    rare_counts = [1, 1, 1, 1, 1, 1]  # 6 labels, sums to 6
    values = []
    for idx, c in enumerate(common_counts):
        values += [f"cat_{idx:02d}"] * c
    for idx, c in enumerate(rare_counts):
        values += [f"cat_{14 + idx:02d}"] * c
    df = pd.DataFrame({"anchor": range(len(values)), "a": values})

    result = clean(df)
    dummy_cols = sorted(c for c in result.columns if c.startswith("a_"))
    assert len(dummy_cols) == 15
    assert "a_other" in dummy_cols
    assert result["a_other"].sum() == 6  # the 6 rare, once-each categories


def test_categorical_dropped_when_too_many_even_for_bucketing():
    values = []
    for i in range(60):
        values += [f"cat_{i:03d}"] * 2
    df = pd.DataFrame({"anchor": range(len(values)), "a": values})
    result = clean(df)
    assert "a" not in result.columns
    assert not any(c.startswith("a_") for c in result.columns)
