import pandas as pd
import pytest

from data_sweep.entity_leakage.keys import score_candidate_keys
from tests.synthetic import make_disjoint_split, make_leaky_split, make_no_entity_structure


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

    # ratio just above 0.95, on a dataset large enough that small-dataset
    # ceiling widening doesn't apply (that's covered separately below)
    df_high = pd.DataFrame({"a": [f"v{i}" for i in range(480)] + [f"v{i}" for i in range(20)]})  # 480/500 = 0.96
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


def test_synthetic_leaky_split_detected_on_test_side_too():
    # inference has to work on whichever file it's run against, not just train
    _, test_df = make_leaky_split(seed=0)
    candidates = score_candidate_keys(test_df)
    assert "entity_id" in {c.column for c in candidates}


def test_synthetic_anonymized_entity_column_still_detected():
    # PRD's core claim: works with zero naming hints. Use a name with no
    # id/key/etc keyword, and values short enough ("E7", "T42") to also miss
    # every format-signal pattern -- detection here can only come from
    # uniqueness_ratio itself, nothing else.
    train_df, test_df = make_leaky_split(entity_col="x7", seed=0)

    train_candidates = score_candidate_keys(train_df)
    by_col = {c.column: c for c in train_candidates}
    assert "x7" in by_col
    assert by_col["x7"].signals == ["uniqueness"]  # no name or format boost fired

    test_candidates = score_candidate_keys(test_df)
    assert "x7" in {c.column for c in test_candidates}


def test_synthetic_disjoint_split_entity_col_still_a_candidate():
    # candidacy is a property of one file's column shape, independent of
    # whether it happens to overlap with another file (that's leakage.py's
    # job, not keys.py's) -- disjoint pools shouldn't stop it being a candidate
    train_df, test_df = make_disjoint_split(seed=0)
    assert "entity_id" in {c.column for c in score_candidate_keys(train_df)}
    assert "entity_id" in {c.column for c in score_candidate_keys(test_df)}


def test_small_dataset_widens_ceiling_above_default_max():
    # 199 rows, 197 unique values -> ratio ~0.99, above the default 0.95
    # ceiling but should still qualify since n_rows is under the threshold
    df = pd.DataFrame({"a": [f"v{i}" for i in range(197)] + ["v0", "v1"]})
    ratio = df["a"].nunique() / len(df)
    assert 0.95 < ratio < 0.99
    candidates = score_candidate_keys(df)
    assert [c.column for c in candidates] == ["a"]
    assert candidates[0].uniqueness_ratio == ratio


def test_small_dataset_ceiling_still_excludes_true_row_ids():
    # every value unique -> ratio 1.0, still excluded even under the
    # widened small-dataset ceiling
    df = pd.DataFrame({"a": [f"v{i}" for i in range(150)]})
    assert score_candidate_keys(df) == []


def test_large_dataset_keeps_default_ceiling():
    # 250 rows (over the small-dataset threshold), ratio 0.97 -> excluded
    # under the normal 0.95 ceiling since widening shouldn't apply here
    df = pd.DataFrame({"a": [f"v{i}" for i in range(242)] + [f"v{i}" for i in range(8)]})
    ratio = df["a"].nunique() / len(df)
    assert 0.95 < ratio < 0.99
    assert score_candidate_keys(df) == []


def test_explicit_max_uniqueness_ratio_overrides_small_dataset_widening():
    # caller-supplied max_uniqueness_ratio should be respected exactly,
    # even on a small dataset, rather than silently widened
    df = pd.DataFrame({"a": [f"v{i}" for i in range(197)] + ["v0", "v1"]})
    assert score_candidate_keys(df, max_uniqueness_ratio=0.95) == []


def test_low_cardinality_column_excluded_even_on_small_dataset():
    # a 3-value categorical can drift into the ratio grouping band purely
    # because the file is small (3/80 = 0.0375, already above the 0.02
    # floor) -- the absolute MIN_UNIQUE_COUNT floor must still exclude it
    df = pd.DataFrame({"status": (["x", "y", "z"] * 27)[:80]})
    assert df["status"].nunique() == 3
    assert score_candidate_keys(df) == []


def test_synthetic_leaky_split_feature_columns_not_flagged():
    # only the real entity column should qualify -- the plain feature columns
    # shouldn't accidentally land in the grouping band
    train_df, _ = make_leaky_split(seed=0)
    candidate_cols = {c.column for c in score_candidate_keys(train_df)}
    assert "feature_a" not in candidate_cols
    assert "feature_b" not in candidate_cols
