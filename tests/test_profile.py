import pandas as pd

from data_sweep.profile import profile


def test_empty_dataframe_produces_no_findings():
    df = pd.DataFrame()
    assert profile(df) == []


def test_single_row_columns_are_constant():
    df = pd.DataFrame({"a": [1], "b": ["x"]})
    findings = profile(df)
    assert len(findings) == 2
    assert {f.issue_type for f in findings} == {"constant_column"}


def test_all_null_column_flagged_constant():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
    findings = [f for f in profile(df) if f.column == "b"]
    assert len(findings) == 1
    assert findings[0].issue_type == "constant_column"
    assert findings[0].confidence == 1.0
    assert "no non-missing values" in findings[0].detail


def test_constant_column_flagged_and_dropped():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "region": ["us", "us", "us", "us"]})
    findings = [f for f in profile(df) if f.column == "region"]
    assert len(findings) == 1
    assert findings[0].issue_type == "constant_column"
    assert findings[0].confidence == 1.0
    assert findings[0].action_taken == "Dropped column."


def test_duplicate_rows_detected():
    df = pd.DataFrame({"a": [1, 2, 1, 3]})
    findings = [f for f in profile(df) if f.issue_type == "duplicate_rows"]
    assert len(findings) == 1
    assert findings[0].column == "(all)"
    assert findings[0].confidence == 1.0
    assert "1 duplicate row" in findings[0].detail


def test_no_duplicate_rows_when_all_unique():
    df = pd.DataFrame({"a": [1, 2, 3]})
    findings = [f for f in profile(df) if f.issue_type == "duplicate_rows"]
    assert findings == []


def test_missing_under_threshold_reports_median_fill():
    df = pd.DataFrame({"a": [10.0, 20.0, 30.0, None]})
    findings = [f for f in profile(df, missing_threshold=0.5) if f.issue_type == "missing_values"]
    assert len(findings) == 1
    assert findings[0].confidence == 1.0
    assert "median" in findings[0].action_taken
    assert "20.0" in findings[0].action_taken


def test_missing_under_threshold_reports_mode_fill_for_text():
    df = pd.DataFrame({"a": ["x", "x", "y", None]})
    findings = [f for f in profile(df, missing_threshold=0.5) if f.issue_type == "missing_values"]
    assert len(findings) == 1
    assert "mode" in findings[0].action_taken


def test_missing_over_threshold_reports_column_drop():
    # anchor column keeps the three all-null rows from collapsing into one
    # via profile()'s own drop_duplicates (NaN == NaN for dedup purposes); "a"
    # itself needs 2+ distinct non-null values so the constant-column check
    # (which runs first) doesn't catch it before the missing-value check does.
    df = pd.DataFrame({"anchor": [1, 2, 3, 4, 5], "a": [1.0, 2.0, None, None, None]})
    findings = [f for f in profile(df, missing_threshold=0.5) if f.issue_type == "missing_values"]
    assert len(findings) == 1
    assert findings[0].action_taken == "Dropped column (too much missing data to impute reliably)."


def test_outliers_detected_on_both_sides_of_iqr():
    df = pd.DataFrame({
        "anchor": range(11),
        "a": [50, 52, 49, 51, 53, 48, 52, 50, 51, 500, -100],
    })
    findings = [f for f in profile(df) if f.issue_type == "outliers"]
    assert len(findings) == 1
    f = findings[0]
    assert f.column == "a"
    assert f.confidence == 0.7
    assert "2 value(s)" in f.detail
    assert "[45.75, 55.75]" in f.detail
    assert "Capped values to [45.75, 55.75]" in f.action_taken


def test_no_outliers_when_values_are_tight():
    df = pd.DataFrame({
        "anchor": range(10),
        "a": [50, 52, 49, 51, 53, 48, 52, 50, 51, 54],
    })
    findings = [f for f in profile(df) if f.issue_type == "outliers"]
    assert findings == []


def test_mixed_type_column_detected_above_threshold():
    # 9/10 values parse as numeric (90%, above the default 0.8 threshold) ->
    # flagged as mixed-type, the stray gets coerced to missing, then the
    # regular missing-value check fires on top with a median fill.
    df = pd.DataFrame({"a": ["10", "11", "12", "13", "14", "15", "16", "17", "18", "unknown"]})
    findings = profile(df)

    mixed = [f for f in findings if f.issue_type == "mixed_type_column"]
    assert len(mixed) == 1
    assert mixed[0].confidence == 0.9
    assert "90% numeric" in mixed[0].detail
    assert "unknown" in mixed[0].detail
    assert mixed[0].action_taken == "Treated unknown as missing and converted column to numeric."

    missing = [f for f in findings if f.issue_type == "missing_values"]
    assert len(missing) == 1
    assert "median (14.0)" in missing[0].action_taken


def test_mixed_type_column_not_flagged_below_threshold():
    # only 5/10 values parse as numeric (50%, below the 0.8 threshold) -> not
    # mixed-type, stays a plain (all-unique) text column and gets dropped as
    # an identifier instead.
    df = pd.DataFrame({"a": ["1", "2", "3", "4", "5", "a", "b", "c", "d", "e"]})
    findings = profile(df)
    assert [f for f in findings if f.issue_type == "mixed_type_column"] == []
    categorical = [f for f in findings if f.issue_type == "categorical_encoding"]
    assert len(categorical) == 1
    assert "identifier" in categorical[0].detail


def test_categorical_ordinal_tier():
    df = pd.DataFrame({"anchor": range(9), "a": ["low", "medium", "high"] * 3})
    findings = [f for f in profile(df) if f.issue_type == "categorical_encoding"]
    assert len(findings) == 1
    assert findings[0].confidence == 0.95
    assert "natural order (low < medium < high)" in findings[0].detail
    assert findings[0].action_taken == "Ordinal encoded using order: low < medium < high."


def test_categorical_one_hot_tier():
    df = pd.DataFrame({"anchor": range(8), "a": ["red", "green", "blue", "yellow"] * 2})
    findings = [f for f in profile(df) if f.issue_type == "categorical_encoding"]
    assert len(findings) == 1
    assert findings[0].confidence == 0.9
    assert "4 unique value(s)" in findings[0].detail
    assert findings[0].action_taken == "One-hot encoded into 4 column(s)."


def test_categorical_identifier_tier():
    df = pd.DataFrame({"anchor": range(10), "a": [f"name_{i}" for i in range(10)]})
    findings = [f for f in profile(df) if f.issue_type == "categorical_encoding"]
    assert len(findings) == 1
    assert findings[0].confidence == 0.85
    assert "identifier or free text" in findings[0].detail
    assert findings[0].action_taken == "Dropped column (values are effectively unique per row, not encodable as categories)."


def test_categorical_bucketed_tier():
    # 20 unique values (above max_categories=15, at or below max_categories_bucketed=50):
    # keep the 14 most frequent, bucket the rest into "other".
    common_counts = [3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2]  # 14 labels, sums to 34
    rare_counts = [1, 1, 1, 1, 1, 1]  # 6 labels, sums to 6
    values = []
    for idx, c in enumerate(common_counts):
        values += [f"cat_{idx:02d}"] * c
    for idx, c in enumerate(rare_counts):
        values += [f"cat_{14 + idx:02d}"] * c
    df = pd.DataFrame({"anchor": range(len(values)), "a": values})

    findings = [f for f in profile(df) if f.issue_type == "categorical_encoding"]
    assert len(findings) == 1
    assert findings[0].confidence == 0.75
    assert "20 unique values" in findings[0].detail
    assert findings[0].action_taken == "Kept the 14 most frequent value(s), bucketed the rest into 'other', then one-hot encoded into 15 column(s)."


def test_categorical_dropped_when_too_many_even_for_bucketing():
    # 60 unique values, each appearing twice (ratio stays at 0.5, so it's not
    # caught by the identifier check) -> above max_categories_bucketed(50), dropped outright.
    values = []
    for i in range(60):
        values += [f"cat_{i:03d}"] * 2
    df = pd.DataFrame({"anchor": range(len(values)), "a": values})

    findings = [f for f in profile(df) if f.issue_type == "categorical_encoding"]
    assert len(findings) == 1
    assert findings[0].confidence == 0.8
    assert "60 unique values" in findings[0].detail
    assert findings[0].action_taken == "Dropped column (too many categories to encode usefully)."


def test_data_leakage_detected_above_threshold():
    df = pd.DataFrame({
        "target": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        "leaky_col": [10, 10, 10, 10, 10, 10, 10, 10, 100, 100],
        "other_col": [5, 3, 8, 1, 9, 4, 7, 2, 6, 10],
    })
    findings = profile(df, target="target")
    leakage = [f for f in findings if f.issue_type == "data_leakage"]
    assert len(leakage) == 1
    assert leakage[0].column == "leaky_col"
    assert leakage[0].confidence == 1.0
    assert "leaky_col' is 100% correlated with target 'target'" in leakage[0].detail
    assert leakage[0].action_taken == "Flagged only (not auto-dropped — review before training on this column)."
    # the uncorrelated column must not be flagged
    assert all(f.column != "other_col" for f in leakage)


def test_no_leakage_below_threshold():
    df = pd.DataFrame({
        "target": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        "other_col": [5, 3, 8, 1, 9, 4, 7, 2, 6, 10],
    })
    findings = [f for f in profile(df, target="target") if f.issue_type == "data_leakage"]
    assert findings == []


def test_no_leakage_when_target_not_numeric():
    df = pd.DataFrame({
        "target": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
        "numeric_col": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    findings = [f for f in profile(df, target="target") if f.issue_type == "data_leakage"]
    assert findings == []


def test_class_imbalance_detected_for_numeric_target():
    # anchor column keeps the 19 repeated "0" rows from collapsing into one
    # via profile()'s own drop_duplicates, which would flip a 95% majority
    # into a 50/50 split.
    df = pd.DataFrame({"anchor": range(20), "target": [0] * 19 + [1]})
    findings = [f for f in profile(df, target="target") if f.issue_type == "class_imbalance"]
    assert len(findings) == 1
    assert findings[0].column == "target"
    assert findings[0].confidence == 0.95
    assert "95% class '0'" in findings[0].detail


def test_class_imbalance_detected_for_text_target():
    df = pd.DataFrame({"anchor": range(20), "target": ["no"] * 19 + ["yes"]})
    findings = [f for f in profile(df, target="target") if f.issue_type == "class_imbalance"]
    assert len(findings) == 1
    assert "95% class 'no'" in findings[0].detail


def test_no_class_imbalance_below_threshold():
    df = pd.DataFrame({"anchor": range(10), "target": [0] * 8 + [1] * 2})  # 80%, below the 90% threshold
    findings = [f for f in profile(df, target="target") if f.issue_type == "class_imbalance"]
    assert findings == []


def test_no_class_imbalance_for_high_cardinality_target():
    # a regression-style target (every value unique) isn't a classification
    # target, so the imbalance check should not apply to it at all.
    df = pd.DataFrame({"target": list(range(20)), "other": list(range(20, 40))})
    findings = [f for f in profile(df, target="target") if f.issue_type == "class_imbalance"]
    assert findings == []


def test_multicollinearity_detected_above_threshold():
    df = pd.DataFrame({
        "col_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "col_b": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        "col_c": [5, 3, 8, 1, 9, 4, 7, 2, 6, 10],
    })
    findings = [f for f in profile(df) if f.issue_type == "multicollinearity"]
    assert len(findings) == 1
    assert findings[0].column == "col_a & col_b"
    assert findings[0].confidence == 1.0
    assert "'col_a' and 'col_b' are 100% correlated" in findings[0].detail
    assert findings[0].action_taken == "Flagged only (consider dropping one of the two)."


def test_no_multicollinearity_below_threshold():
    df = pd.DataFrame({
        "col_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "col_c": [5, 3, 8, 1, 9, 4, 7, 2, 6, 10],
    })
    findings = [f for f in profile(df) if f.issue_type == "multicollinearity"]
    assert findings == []


def test_multicollinearity_excludes_target_column():
    # target itself is perfectly correlated with col_a here, but the
    # multicollinearity screen only compares predictor columns to each
    # other, never to the target (that's leakage's job).
    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "col_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })
    findings = [f for f in profile(df, target="target") if f.issue_type == "multicollinearity"]
    assert findings == []
