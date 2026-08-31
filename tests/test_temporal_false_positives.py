"""False-positive sweep for temporal leakage: does check_temporal_leakage()
stay quiet on legitimately-windowed aggregates and plain unrelated numeric
columns, across many seeds -- not just avoid false positives on one lucky
run?
"""
import numpy as np
import pandas as pd
import pytest

from data_sweep.entity_leakage.temporal import check_temporal_leakage
from tests.synthetic import make_temporal_leak_dataset


@pytest.mark.parametrize("seed", range(20))
def test_all_clean_dataset_never_flags_high(seed):
    # both leak-specific boosts switched off -- both features are now
    # legitimately windowed, ground truth: no leak
    df = make_temporal_leak_dataset(seed=seed, elapsed_leak_strength=0.0, target_leak_boost=0.0)
    findings = check_temporal_leakage(
        df, target_col="target", event_time_col="cancel_date", record_time_col="snapshot_date",
    )
    assert all(f.severity != "HIGH" for f in findings), f"seed {seed}: clean dataset produced a HIGH finding"


@pytest.mark.parametrize("seed", range(20))
def test_random_unrelated_columns_produce_no_findings(seed):
    # plain independent noise, naming-neutral columns, no timestamps --
    # nothing here should ever fire any of the three signals
    rng = np.random.RandomState(seed)
    n = 500
    df = pd.DataFrame({
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.normal(0, 1, n),
        "feature_c": rng.uniform(0, 100, n),
        "target": rng.randint(0, 2, n),
    })
    findings = check_temporal_leakage(df, target_col="target")
    assert findings == [], f"seed {seed}: unrelated random data produced {len(findings)} finding(s)"


@pytest.mark.parametrize("seed", range(20))
def test_random_unrelated_columns_with_timestamps_produce_no_findings(seed):
    # same as above but with real event/record timestamps supplied -- signal
    # 2 gets a genuine chance to fire on pure noise and shouldn't
    rng = np.random.RandomState(seed)
    n = 500
    record_time = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.uniform(0, 365, n), unit="D")
    event_time = record_time + pd.to_timedelta(rng.uniform(1, 200, n), unit="D")
    df = pd.DataFrame({
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.normal(0, 1, n),
        "snapshot_date": record_time,
        "cancel_date": event_time,
        "target": rng.randint(0, 2, n),
    })
    findings = check_temporal_leakage(
        df, target_col="target", event_time_col="cancel_date", record_time_col="snapshot_date",
    )
    assert findings == [], f"seed {seed}: unrelated random data (with timestamps) produced a finding"


@pytest.mark.parametrize("seed", range(10))
def test_aggregation_named_but_genuinely_clean_column_stays_low(seed):
    # a column that matches the name signal but has no real relationship to
    # elapsed time or the target -- name alone should cap it at LOW, never
    # MEDIUM or HIGH
    rng = np.random.RandomState(seed)
    n = 500
    record_time = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.uniform(0, 365, n), unit="D")
    event_time = record_time + pd.to_timedelta(rng.uniform(1, 200, n), unit="D")
    df = pd.DataFrame({
        "total_widgets": rng.normal(50, 10, n),  # matches "total_" but is pure noise
        "snapshot_date": record_time,
        "cancel_date": event_time,
        "target": rng.randint(0, 2, n),
    })
    findings = check_temporal_leakage(
        df, target_col="target", event_time_col="cancel_date", record_time_col="snapshot_date",
    )
    by_feature = {f.feature: f for f in findings}
    if "total_widgets" in by_feature:
        assert by_feature["total_widgets"].severity == "LOW"
