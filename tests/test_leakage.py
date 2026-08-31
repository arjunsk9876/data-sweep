import pandas as pd

from data_sweep.entity_leakage.keys import CandidateKey
from data_sweep.entity_leakage.leakage import LeakageFinding, check_cross_split_leakage, rank_by_severity
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


def test_multiple_leaks_ranked_by_overlap_ratio_descending():
    # both columns: 100 train entities x 4 rows = 400 rows
    train_df = pd.DataFrame({
        "col_low": [f"L{i}" for i in range(100) for _ in range(4)],
        "col_high": [f"H{i}" for i in range(100) for _ in range(4)],
    })
    # col_low test: 5 shared (L0-L4) + 95 disjoint -> 5/100 = 5% overlap
    # col_high test: 50 shared (H0-H49) + 50 disjoint -> 50/100 = 50% overlap
    test_df = pd.DataFrame({
        "col_low": [f"L{i}" for i in range(5) for _ in range(4)] + [f"M{i}" for i in range(95) for _ in range(4)],
        "col_high": [f"H{i}" for i in range(50) for _ in range(4)] + [f"N{i}" for i in range(50) for _ in range(4)],
    })
    assert len(train_df) == 400 and len(test_df) == 400

    findings = check_cross_split_leakage(train_df, test_df, overlap_threshold=0.0)
    assert [f.column for f in findings] == ["col_high", "col_low"]
    assert findings[0].overlap_ratio > findings[1].overlap_ratio


def test_severity_tiebreak_uses_overlap_count():
    # both findings: 50% overlap ratio, but col_a affects far more entities
    findings = [
        _fake_finding("col_a", overlap_ratio=0.5, overlap_count=50, test_entity_count=100, score=1.0),
        _fake_finding("col_b", overlap_ratio=0.5, overlap_count=10, test_entity_count=20, score=1.0),
    ]
    ranked = rank_by_severity(findings)
    assert [f.column for f in ranked] == ["col_a", "col_b"]


def _fake_finding(column, overlap_ratio, overlap_count, test_entity_count, score):
    return LeakageFinding(
        column=column,
        overlap_ratio=overlap_ratio,
        overlap_count=overlap_count,
        test_entity_count=test_entity_count,
        candidate_key=CandidateKey(column=column, uniqueness_ratio=0.5, score=score, signals=["uniqueness"]),
        example_overlapping_values=[],
    )
