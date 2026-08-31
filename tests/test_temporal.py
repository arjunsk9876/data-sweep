import pandas as pd
import pytest

from data_sweep.entity_leakage.temporal import compute_elapsed_time_correlation, detect_aggregation_name_signal
from tests.synthetic import make_temporal_leak_dataset


def test_total_prefix_matches():
    assert detect_aggregation_name_signal("total_purchases") is True


def test_avg_prefix_matches():
    assert detect_aggregation_name_signal("avg_response_time") is True


def test_cumulative_prefix_matches():
    assert detect_aggregation_name_signal("cumulative_spend") is True


def test_lifetime_prefix_matches():
    assert detect_aggregation_name_signal("lifetime_value") is True


def test_running_prefix_matches():
    assert detect_aggregation_name_signal("running_total") is True


def test_ytd_prefix_matches():
    assert detect_aggregation_name_signal("ytd_revenue") is True


def test_sum_mean_count_last_max_min_all_match():
    assert detect_aggregation_name_signal("sum_orders") is True
    assert detect_aggregation_name_signal("mean_basket_size") is True
    assert detect_aggregation_name_signal("count_logins") is True
    assert detect_aggregation_name_signal("last_purchase_amount") is True
    assert detect_aggregation_name_signal("max_order_value") is True
    assert detect_aggregation_name_signal("min_order_value") is True


def test_keyword_matches_mid_name_not_just_prefix():
    assert detect_aggregation_name_signal("customer_total_purchases") is True


def test_to_date_suffix_matches():
    assert detect_aggregation_name_signal("purchases_to_date") is True


def test_matches_both_leaked_and_legitimately_windowed_names():
    # naming alone can't discriminate leaked from clean -- both should fire
    assert detect_aggregation_name_signal("total_purchases") is True
    assert detect_aggregation_name_signal("total_purchases_windowed") is True


def test_climax_does_not_false_positive_on_max():
    assert detect_aggregation_name_signal("climax_score") is False


def test_update_date_does_not_false_positive_on_to_date():
    assert detect_aggregation_name_signal("update_date") is False


def test_maximum_does_not_false_positive_on_max():
    assert detect_aggregation_name_signal("maximum_capacity") is False


def test_unrelated_column_name_does_not_match():
    assert detect_aggregation_name_signal("customer_id") is False
    assert detect_aggregation_name_signal("region") is False


def test_custom_keywords_list_respected():
    assert detect_aggregation_name_signal("weird_score", keywords=["weird"]) is True
    assert detect_aggregation_name_signal("total_purchases", keywords=["weird"]) is False


def test_leaked_feature_has_strong_positive_elapsed_correlation():
    df = make_temporal_leak_dataset(seed=0)
    corr = compute_elapsed_time_correlation(df["total_purchases"], df["snapshot_date"], df["cancel_date"])
    assert corr is not None
    assert corr > 0.3


def test_clean_feature_has_near_zero_elapsed_correlation():
    df = make_temporal_leak_dataset(seed=0)
    corr = compute_elapsed_time_correlation(df["total_purchases_windowed"], df["snapshot_date"], df["cancel_date"])
    assert corr is not None
    assert abs(corr) < 0.15


def test_perfectly_linear_relationship_gives_correlation_near_one():
    record_time = pd.to_datetime(["2023-01-01"] * 100)
    event_time = record_time + pd.to_timedelta(range(100), unit="D")
    feature = pd.Series(range(100), dtype=float)
    corr = compute_elapsed_time_correlation(feature, record_time, event_time)
    assert corr is not None
    assert corr == pytest.approx(1.0)


def test_inverse_relationship_gives_negative_correlation():
    # signed, not direction-agnostic -- a feature that shrinks as elapsed
    # time grows tells a different story than one that grows, so the sign
    # must be preserved rather than folded into an absolute value
    record_time = pd.to_datetime(["2023-01-01"] * 100)
    event_time = record_time + pd.to_timedelta(range(100), unit="D")
    feature = pd.Series(range(100, 0, -1), dtype=float)
    corr = compute_elapsed_time_correlation(feature, record_time, event_time)
    assert corr is not None
    assert corr < -0.9


def test_constant_feature_returns_none():
    record_time = pd.to_datetime(["2023-01-01"] * 50)
    event_time = record_time + pd.to_timedelta(range(50), unit="D")
    feature = pd.Series([7.0] * 50)
    assert compute_elapsed_time_correlation(feature, record_time, event_time) is None


def test_constant_elapsed_time_returns_none():
    # every row has the same gap -- nothing to correlate against
    record_time = pd.to_datetime(["2023-01-01"] * 50)
    event_time = pd.to_datetime(["2023-02-01"] * 50)
    feature = pd.Series(range(50), dtype=float)
    assert compute_elapsed_time_correlation(feature, record_time, event_time) is None


def test_too_few_rows_returns_none():
    record_time = pd.to_datetime(["2023-01-01"] * 5)
    event_time = record_time + pd.to_timedelta(range(5), unit="D")
    feature = pd.Series(range(5), dtype=float)
    assert compute_elapsed_time_correlation(feature, record_time, event_time) is None


def test_missing_values_are_dropped_before_correlating():
    record_time = pd.to_datetime(["2023-01-01"] * 30)
    event_time = record_time + pd.to_timedelta(range(30), unit="D")
    feature = pd.Series(list(range(25)) + [None] * 5, dtype=float)
    corr = compute_elapsed_time_correlation(feature, record_time, event_time)
    assert corr is not None  # shouldn't crash or return None just because some rows are missing
