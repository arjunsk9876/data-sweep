import pandas as pd

from data_sweep.entity_leakage.leakage import check_cross_split_leakage
from tests.synthetic import make_disjoint_split, make_leaky_split, make_no_entity_structure


def test_leaky_split_detected():
    train_df, test_df = make_leaky_split(overlap_entities=20, seed=0)
    findings = check_cross_split_leakage(train_df, test_df)

    by_col = {f.column: f for f in findings}
    assert "entity_id" in by_col
    f = by_col["entity_id"]
    assert f.overlap_count == 20  # exact, guaranteed by the generator
    assert 0 < f.overlap_ratio <= 1
    assert f.test_entity_count >= f.overlap_count
    assert len(f.example_overlapping_values) <= 3
    assert f.candidate_key.column == "entity_id"


def test_disjoint_split_finds_no_leakage():
    train_df, test_df = make_disjoint_split(seed=0)
    findings = check_cross_split_leakage(train_df, test_df)
    assert findings == []


def test_no_entity_structure_finds_no_leakage():
    train_df, test_df = make_no_entity_structure(seed=0)
    findings = check_cross_split_leakage(train_df, test_df)
    assert findings == []


def test_column_missing_from_test_is_skipped_not_errored():
    train_df, test_df = make_leaky_split(seed=0)
    test_df = test_df.drop(columns=["entity_id"])
    findings = check_cross_split_leakage(train_df, test_df)
    assert findings == []


def test_empty_test_values_skipped():
    train_df, _ = make_leaky_split(seed=0)
    test_df = pd.DataFrame({"entity_id": [], "feature_a": [], "feature_b": []})
    findings = check_cross_split_leakage(train_df, test_df)
    assert findings == []


def test_overlap_below_threshold_not_flagged():
    # 1 shared entity out of a large test pool -> overlap ratio well under 2%
    train_df, test_df = make_leaky_split(
        n_train=1000, n_test=1000,
        n_entities_train=500, n_entities_test_only=500,
        overlap_entities=1, seed=0,
    )
    findings = check_cross_split_leakage(train_df, test_df)
    assert findings == []


def test_overlap_above_threshold_flagged_with_custom_threshold():
    train_df, test_df = make_leaky_split(
        n_train=1000, n_test=1000,
        n_entities_train=500, n_entities_test_only=500,
        overlap_entities=1, seed=0,
    )
    # with a very low threshold, even a single shared entity should trigger
    findings = check_cross_split_leakage(train_df, test_df, overlap_threshold=0.0001)
    assert len(findings) == 1
    assert findings[0].column == "entity_id"


def test_example_overlapping_values_are_real_overlaps():
    train_df, test_df = make_leaky_split(overlap_entities=20, seed=0)
    findings = check_cross_split_leakage(train_df, test_df)
    f = next(f for f in findings if f.column == "entity_id")

    train_values = set(train_df["entity_id"])
    test_values = set(test_df["entity_id"])
    for example in f.example_overlapping_values:
        assert example in train_values
        assert example in test_values
