import pandas as pd
import pytest

from data_sweep.entity_leakage.baseline import PredictivenessResult
from data_sweep.entity_leakage.temporal import (
    check_temporal_leakage,
    combine_temporal_signals,
    compute_elapsed_time_correlation,
    compute_predictiveness_signal,
    detect_aggregation_name_signal,
    is_unusually_predictive,
)
from tests.synthetic import make_temporal_leak_dataset

_HIGH_PREDICTIVENESS = PredictivenessResult(score=0.9, metric="auc")
_LOW_PREDICTIVENESS = PredictivenessResult(score=0.55, metric="auc")


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


def test_leaked_feature_scores_unusually_predictive():
    df = make_temporal_leak_dataset(seed=0)
    result = compute_predictiveness_signal(df["total_purchases"], df["target"])
    assert result is not None
    assert is_unusually_predictive(result) is True


def test_clean_feature_does_not_score_unusually_predictive():
    df = make_temporal_leak_dataset(seed=0)
    result = compute_predictiveness_signal(df["total_purchases_windowed"], df["target"])
    assert result is not None
    assert is_unusually_predictive(result) is False


def test_is_unusually_predictive_boundary():
    assert is_unusually_predictive(PredictivenessResult(score=0.8, metric="auc")) is True
    assert is_unusually_predictive(PredictivenessResult(score=0.799999, metric="auc")) is False


def test_is_unusually_predictive_respects_custom_threshold():
    result = PredictivenessResult(score=0.6, metric="auc")
    assert is_unusually_predictive(result, threshold=0.5) is True
    assert is_unusually_predictive(result, threshold=0.7) is False


def test_is_unusually_predictive_none_is_never_suspicious():
    assert is_unusually_predictive(None) is False


def test_combine_signals_none_fired_gives_no_finding():
    assert combine_temporal_signals(False, None, None) is None
    assert combine_temporal_signals(False, 0.1, _LOW_PREDICTIVENESS) is None


def test_combine_signals_one_fired_gives_low():
    assert combine_temporal_signals(True, None, None) == "LOW"
    assert combine_temporal_signals(False, 0.5, None) == "LOW"
    assert combine_temporal_signals(False, None, _HIGH_PREDICTIVENESS) == "LOW"


def test_combine_signals_two_fired_gives_medium():
    assert combine_temporal_signals(True, 0.5, None) == "MEDIUM"
    assert combine_temporal_signals(True, None, _HIGH_PREDICTIVENESS) == "MEDIUM"
    assert combine_temporal_signals(False, 0.5, _HIGH_PREDICTIVENESS) == "MEDIUM"


def test_combine_signals_three_fired_gives_high():
    assert combine_temporal_signals(True, 0.5, _HIGH_PREDICTIVENESS) == "HIGH"


def test_combine_signals_weak_correlation_does_not_count_as_fired():
    # below ELAPSED_CORRELATION_STRONG_THRESHOLD -- shouldn't push a single
    # name match up to two signals fired
    assert combine_temporal_signals(True, 0.1, None) == "LOW"


def test_combine_signals_low_predictiveness_does_not_count_as_fired():
    assert combine_temporal_signals(True, None, _LOW_PREDICTIVENESS) == "LOW"


def test_combine_signals_matches_synthetic_leaked_and_clean_features():
    df = make_temporal_leak_dataset(seed=0)

    leaked_name = detect_aggregation_name_signal("total_purchases")
    leaked_corr = compute_elapsed_time_correlation(df["total_purchases"], df["snapshot_date"], df["cancel_date"])
    leaked_pred = compute_predictiveness_signal(df["total_purchases"], df["target"])
    assert combine_temporal_signals(leaked_name, leaked_corr, leaked_pred) == "HIGH"

    clean_name = detect_aggregation_name_signal("total_purchases_windowed")
    clean_corr = compute_elapsed_time_correlation(df["total_purchases_windowed"], df["snapshot_date"], df["cancel_date"])
    clean_pred = compute_predictiveness_signal(df["total_purchases_windowed"], df["target"])
    # the name still matches (naming alone doesn't discriminate), but
    # neither of the other two signals should fire on a clean feature
    assert clean_name is True
    assert combine_temporal_signals(clean_name, clean_corr, clean_pred) == "LOW"


def test_check_temporal_leakage_full_confidence_flags_leaked_feature():
    df = make_temporal_leak_dataset(seed=0)
    findings = check_temporal_leakage(
        df, target_col="target", event_time_col="cancel_date", record_time_col="snapshot_date",
    )
    by_feature = {f.feature: f for f in findings}
    assert "total_purchases" in by_feature
    leaked = by_feature["total_purchases"]
    assert leaked.severity == "HIGH"
    assert leaked.reduced_confidence is False
    assert leaked.elapsed_time_correlation is not None
    assert leaked.predictiveness_score is not None


def test_check_temporal_leakage_full_confidence_does_not_flag_clean_feature():
    df = make_temporal_leak_dataset(seed=0)
    findings = check_temporal_leakage(
        df, target_col="target", event_time_col="cancel_date", record_time_col="snapshot_date",
    )
    by_feature = {f.feature: f for f in findings}
    # the clean control may still show up at LOW (name signal alone fires)
    # but must never reach MEDIUM/HIGH like the leaked feature does
    if "total_purchases_windowed" in by_feature:
        assert by_feature["total_purchases_windowed"].severity == "LOW"


def test_check_temporal_leakage_excludes_target_and_timestamp_columns():
    df = make_temporal_leak_dataset(seed=0)
    findings = check_temporal_leakage(
        df, target_col="target", event_time_col="cancel_date", record_time_col="snapshot_date",
    )
    flagged_features = {f.feature for f in findings}
    assert "target" not in flagged_features
    assert "cancel_date" not in flagged_features
    assert "snapshot_date" not in flagged_features


def test_check_temporal_leakage_reduced_confidence_without_timestamps():
    df = make_temporal_leak_dataset(seed=0)
    findings = check_temporal_leakage(df, target_col="target")
    by_feature = {f.feature: f for f in findings}
    assert "total_purchases" in by_feature
    leaked = by_feature["total_purchases"]
    assert leaked.reduced_confidence is True
    assert leaked.elapsed_time_correlation is None  # signal 2 never ran
    # only 2 signals possible now (name + predictiveness), so HIGH (which
    # needs all 3) is unreachable -- MEDIUM is the ceiling
    assert leaked.severity in ("LOW", "MEDIUM")


def test_check_temporal_leakage_reduced_confidence_with_only_one_timestamp():
    # both are required for signal 2 -- providing just one should behave
    # exactly like providing neither
    df = make_temporal_leak_dataset(seed=0)
    findings = check_temporal_leakage(df, target_col="target", event_time_col="cancel_date")
    by_feature = {f.feature: f for f in findings}
    assert by_feature["total_purchases"].reduced_confidence is True
    assert by_feature["total_purchases"].elapsed_time_correlation is None


def test_check_temporal_leakage_all_clean_dataset_produces_few_or_no_high_findings():
    df = make_temporal_leak_dataset(seed=0, elapsed_leak_strength=0.0, target_leak_boost=0.0)
    findings = check_temporal_leakage(
        df, target_col="target", event_time_col="cancel_date", record_time_col="snapshot_date",
    )
    assert all(f.severity != "HIGH" for f in findings)


def test_check_temporal_leakage_findings_ranked_worst_first():
    df = make_temporal_leak_dataset(seed=0)
    # add a second, medium-severity feature: matches the name pattern and
    # correlates with elapsed time, but isn't boosted by target directly
    df["avg_session_length"] = df["total_purchases_windowed"] + 0.5 * (
        (df["cancel_date"] - df["snapshot_date"]).dt.total_seconds() / 86400
    )
    findings = check_temporal_leakage(
        df, target_col="target", event_time_col="cancel_date", record_time_col="snapshot_date",
    )
    severities = [f.severity for f in findings]
    ranks = [{"HIGH": 3, "MEDIUM": 2, "LOW": 1}[s] for s in severities]
    assert ranks == sorted(ranks, reverse=True)
