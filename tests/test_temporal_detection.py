"""Detection sweep: does check_temporal_leakage() reliably catch the
injected leak across many seeds and shapes, not just one lucky run?

Complements the single hand-picked-seed tests in test_temporal.py --
those prove the mechanism works; this proves it isn't a fluke of seed=0.
"""
import pytest

from data_sweep.entity_leakage.temporal import check_temporal_leakage
from tests.synthetic import make_temporal_leak_dataset


@pytest.mark.parametrize("seed", range(20))
def test_leaked_feature_detected_across_seeds(seed):
    df = make_temporal_leak_dataset(seed=seed)
    findings = check_temporal_leakage(
        df, target_col="target", event_time_col="cancel_date", record_time_col="snapshot_date",
    )
    by_feature = {f.feature: f for f in findings}
    assert "total_purchases" in by_feature, f"seed {seed}: leaked feature not flagged at all"
    assert by_feature["total_purchases"].severity == "HIGH", f"seed {seed}: leaked feature not HIGH"


@pytest.mark.parametrize("seed", range(10))
def test_leaked_feature_detected_at_smaller_scale(seed):
    # at n=200, sampling noise can legitimately soften one of the three
    # signals below its threshold for a given seed -- still correctly
    # flagged, just not guaranteed to reach the same HIGH severity a
    # larger, less noisy sample would
    df = make_temporal_leak_dataset(n_rows=200, seed=seed)
    findings = check_temporal_leakage(
        df, target_col="target", event_time_col="cancel_date", record_time_col="snapshot_date",
    )
    by_feature = {f.feature: f for f in findings}
    assert "total_purchases" in by_feature
    assert by_feature["total_purchases"].severity in ("MEDIUM", "HIGH")


def test_leaked_feature_detected_at_larger_scale():
    df = make_temporal_leak_dataset(n_rows=5000, seed=1)
    findings = check_temporal_leakage(
        df, target_col="target", event_time_col="cancel_date", record_time_col="snapshot_date",
    )
    by_feature = {f.feature: f for f in findings}
    assert "total_purchases" in by_feature
    assert by_feature["total_purchases"].severity == "HIGH"


@pytest.mark.parametrize("seed", range(10))
def test_leaked_feature_flagged_even_without_a_matching_name(seed):
    # rename the leaked column to something that doesn't match any
    # aggregation keyword -- only Signal 2 and Signal 3 can fire now, so
    # severity should cap at MEDIUM, but it must still be flagged
    df = make_temporal_leak_dataset(seed=seed)
    df = df.rename(columns={"total_purchases": "x9"})
    findings = check_temporal_leakage(
        df, target_col="target", event_time_col="cancel_date", record_time_col="snapshot_date",
    )
    by_feature = {f.feature: f for f in findings}
    assert "x9" in by_feature, f"seed {seed}: anonymized leaked feature not flagged"
    assert by_feature["x9"].severity in ("MEDIUM", "HIGH")
    assert by_feature["x9"].name_signal_matched is False
