import pandas as pd
import pytest

from data_sweep.entity_leakage.keys import score_candidate_keys
from tests.synthetic import make_leaky_split, make_no_entity_structure


def test_empty_dataframe_produces_no_candidates():
    assert score_candidate_keys(pd.DataFrame()) == []


def test_row_unique_column_excluded():
    # ~100% unique -> a true row id, not a grouping key
    df = pd.DataFrame({"row_id": [f"r{i}" for i in range(200)]})
    candidates = score_candidate_keys(df)
    assert [c.column for c in candidates] == []


def test_low_cardinality_column_excluded():
    # 3 unique values across 1000 rows -> a plain categorical, not a grouping key
    df = pd.DataFrame({"category": (["a", "b", "c"] * 400)[:1000]})
    candidates = score_candidate_keys(df)
    assert [c.column for c in candidates] == []


def test_grouping_band_column_included():
    # 50 entities across 500 rows -> ratio 0.1, squarely in the grouping band
    # (name deliberately has no id/key/etc keyword, to isolate the uniqueness signal)
    df = pd.DataFrame({"widget_ref": [f"h{i % 50}" for i in range(500)]})
    candidates = score_candidate_keys(df)
    assert len(candidates) == 1
    assert candidates[0].column == "widget_ref"
    assert candidates[0].uniqueness_ratio == 0.1
    assert candidates[0].signals == ["uniqueness"]


def test_format_signal_boosts_score_and_ranks_above_plain_uniqueness():
    # column names deliberately avoid id/key/etc keywords, to isolate the format signal
    df = pd.DataFrame({
        # 50 entities across 500 rows, zero-padded -> uniqueness + format
        "col_zeropad": [f"{i % 50:05d}" for i in range(500)],
        # same grouping-band shape, plain digits -> uniqueness only
        "col_plain": [f"{i % 50}" for i in range(500)],
    })
    candidates = score_candidate_keys(df)
    by_col = {c.column: c for c in candidates}

    assert by_col["col_zeropad"].signals == ["uniqueness", "format"]
    assert by_col["col_plain"].signals == ["uniqueness"]
    assert by_col["col_zeropad"].score > by_col["col_plain"].score
    # format signal should rank the ID-formatted column first
    assert candidates[0].column == "col_zeropad"


def test_name_signal_boosts_score_and_ranks_above_plain_uniqueness():
    # values are plain (no format signal either way) so this isolates the name signal
    df = pd.DataFrame({
        # keyword in the name -> uniqueness + name
        "household_id": [f"{i % 50}" for i in range(500)],
        # same grouping-band shape, neutral name -> uniqueness only
        "widget_ref": [f"{i % 50}" for i in range(500)],
    })
    candidates = score_candidate_keys(df)
    by_col = {c.column: c for c in candidates}

    assert by_col["household_id"].signals == ["uniqueness", "name"]
    assert by_col["widget_ref"].signals == ["uniqueness"]
    assert by_col["household_id"].score > by_col["widget_ref"].score
    assert candidates[0].column == "household_id"


def test_all_three_signals_compound():
    # zero-padded values + id-hinting name -> uniqueness + format + name
    df = pd.DataFrame({"customer_id": [f"{i % 50:05d}" for i in range(500)]})
    candidates = score_candidate_keys(df)
    assert candidates[0].signals == ["uniqueness", "format", "name"]
    assert candidates[0].score == pytest.approx(1.0 + 0.15 + 0.1)


def test_boundary_ratios_are_inclusive():
    # ratio exactly 0.02 (10 uniques / 500 rows)
    df_low = pd.DataFrame({"a": [f"v{i % 10}" for i in range(500)]})
    assert score_candidate_keys(df_low)[0].uniqueness_ratio == 0.02

    # ratio exactly 0.95 (95 uniques / 100 rows... need integer row/unique counts)
    df_high = pd.DataFrame({"a": [f"v{i}" for i in range(95)] + [f"v{i}" for i in range(5)]})
    ratio = df_high["a"].nunique() / len(df_high)
    assert ratio == 0.95
    assert score_candidate_keys(df_high)[0].uniqueness_ratio == 0.95


def test_just_outside_boundary_excluded():
    # ratio just below 0.02
    df_low = pd.DataFrame({"a": [f"v{i % 9}" for i in range(500)]})  # 9/500 = 0.018
    assert score_candidate_keys(df_low) == []

    # ratio just above 0.95
    df_high = pd.DataFrame({"a": [f"v{i}" for i in range(96)] + [f"v0" for _ in range(4)]})  # 96/100 = 0.96
    assert score_candidate_keys(df_high) == []


def test_multiple_candidates_all_returned():
    df = pd.DataFrame({
        "household_id": [f"h{i % 50}" for i in range(500)],
        "device_id": [f"d{i % 100}" for i in range(500)],
        "row_id": [f"r{i}" for i in range(500)],
    })
    candidates = score_candidate_keys(df)
    assert {c.column for c in candidates} == {"household_id", "device_id"}


def test_synthetic_leaky_split_entity_col_detected():
    train_df, _ = make_leaky_split(seed=0)
    candidates = score_candidate_keys(train_df)
    assert "entity_id" in {c.column for c in candidates}


def test_synthetic_no_entity_structure_finds_no_candidates():
    train_df, _ = make_no_entity_structure(seed=0)
    assert score_candidate_keys(train_df) == []
