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
    # needs 2+ distinct non-null values, otherwise the constant-column check
    # (which runs first) catches it before the missing-value check does.
    df = pd.DataFrame({"a": [1.0, 2.0, None, None, None]})
    findings = [f for f in profile(df, missing_threshold=0.5) if f.issue_type == "missing_values"]
    assert len(findings) == 1
    assert findings[0].action_taken == "Dropped column (too much missing data to impute reliably)."
