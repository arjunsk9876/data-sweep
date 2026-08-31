import pandas as pd

from data_sweep.entity_leakage.baseline import compute_predictiveness
from tests.synthetic import (
    make_disjoint_split,
    make_leaky_split,
    make_no_entity_structure,
    make_temporal_leak_dataset,
)


def _elapsed_days(df):
    return (df["cancel_date"] - df["snapshot_date"]).dt.total_seconds() / 86400


def test_leaky_split_has_exact_injected_overlap():
    train_df, test_df = make_leaky_split(overlap_entities=20, n_entities_train=100, n_entities_test_only=100, seed=0)
    train_entities = set(train_df["entity_id"])
    test_entities = set(test_df["entity_id"])
    assert len(train_entities & test_entities) == 20


def test_leaky_split_entities_are_not_row_unique():
    # each entity should appear multiple times, not once per row
    train_df, _ = make_leaky_split(n_train=500, n_entities_train=100, seed=0)
    unique_ratio = train_df["entity_id"].nunique() / len(train_df)
    assert 0.02 < unique_ratio < 0.95


def test_disjoint_split_has_zero_overlap():
    train_df, test_df = make_disjoint_split(seed=0)
    train_entities = set(train_df["entity_id"])
    test_entities = set(test_df["entity_id"])
    assert len(train_entities & test_entities) == 0


def test_disjoint_split_entities_are_not_row_unique():
    train_df, _ = make_disjoint_split(n_train=500, n_entities_train=100, seed=0)
    unique_ratio = train_df["entity_id"].nunique() / len(train_df)
    assert 0.02 < unique_ratio < 0.95


def test_no_entity_structure_has_no_grouping_band_column():
    train_df, test_df = make_no_entity_structure(seed=0)
    for col in train_df.columns:
        ratio = train_df[col].nunique() / len(train_df)
        assert ratio <= 0.02 or ratio >= 0.95, f"{col} unexpectedly falls in the grouping band ({ratio:.2f})"


def test_no_entity_structure_row_ids_are_disjoint_across_splits():
    train_df, test_df = make_no_entity_structure(seed=0)
    assert set(train_df["row_id"]) & set(test_df["row_id"]) == set()


def test_generators_are_deterministic_given_same_seed():
    train_a, test_a = make_leaky_split(seed=42)
    train_b, test_b = make_leaky_split(seed=42)
    assert train_a["entity_id"].tolist() == train_b["entity_id"].tolist()
    assert test_a["entity_id"].tolist() == test_b["entity_id"].tolist()


def test_temporal_leak_dataset_has_expected_columns():
    df = make_temporal_leak_dataset(seed=0)
    assert set(df.columns) == {
        "snapshot_date", "cancel_date", "target", "total_purchases", "total_purchases_windowed",
    }


def test_temporal_leak_dataset_deterministic_given_same_seed():
    a = make_temporal_leak_dataset(seed=7)
    b = make_temporal_leak_dataset(seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_temporal_leak_dataset_target_is_binary():
    df = make_temporal_leak_dataset(seed=0)
    assert df["target"].nunique() == 2
    assert set(df["target"].unique()) <= {0, 1}


def test_temporal_leak_dataset_cancel_date_after_snapshot_date():
    df = make_temporal_leak_dataset(seed=0)
    assert (df["cancel_date"] > df["snapshot_date"]).all()


def test_leaked_feature_correlates_with_elapsed_time():
    # the whole point of the leak: its value grows with how much history
    # extended past the label event, which a correctly-windowed feature
    # should never do
    df = make_temporal_leak_dataset(seed=0)
    corr = df["total_purchases"].corr(_elapsed_days(df))
    assert corr > 0.3


def test_clean_control_feature_does_not_correlate_with_elapsed_time():
    df = make_temporal_leak_dataset(seed=0)
    corr = df["total_purchases_windowed"].corr(_elapsed_days(df))
    assert abs(corr) < 0.15


def test_leaked_feature_is_more_predictive_than_clean_control():
    df = make_temporal_leak_dataset(seed=0)
    leaked_result = compute_predictiveness(df["total_purchases"], df["target"])
    clean_result = compute_predictiveness(df["total_purchases_windowed"], df["target"])
    assert leaked_result is not None and clean_result is not None
    assert leaked_result.score > clean_result.score
    assert leaked_result.score > 0.75  # unusually high for a single feature


def test_zero_strength_produces_an_all_clean_dataset():
    df = make_temporal_leak_dataset(seed=0, elapsed_leak_strength=0.0, target_leak_boost=0.0)

    assert abs(df["total_purchases"].corr(_elapsed_days(df))) < 0.15
    assert abs(df["total_purchases_windowed"].corr(_elapsed_days(df))) < 0.15

    leaked_result = compute_predictiveness(df["total_purchases"], df["target"])
    clean_result = compute_predictiveness(df["total_purchases_windowed"], df["target"])
    assert leaked_result is not None and clean_result is not None
    # both features are now comparably (mildly) predictive -- neither is
    # unusually so, since the leak-specific boosts are both switched off
    assert leaked_result.score < 0.75
    assert clean_result.score < 0.75
