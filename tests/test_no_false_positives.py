"""Statistical false-positive sweep for the entity-leakage detector.

The targeted tests in test_leakage.py and test_keys.py each check one
fixed scenario. This module instead sweeps many seeds and shapes across
the "ground truth: no leak" generators, since a single lucky/unlucky seed
could hide a rare false positive that a scoring or threshold bug would
introduce.
"""
import pandas as pd
import pytest

from data_sweep.entity_leakage.leakage import check_cross_split_leakage
from tests.synthetic import make_disjoint_split, make_no_entity_structure

SEEDS = range(30)


@pytest.mark.parametrize("seed", SEEDS)
def test_disjoint_split_never_flagged_across_seeds(seed):
    train_df, test_df = make_disjoint_split(seed=seed)
    assert check_cross_split_leakage(train_df, test_df) == []


@pytest.mark.parametrize("seed", SEEDS)
def test_no_entity_structure_never_flagged_across_seeds(seed):
    train_df, test_df = make_no_entity_structure(seed=seed)
    assert check_cross_split_leakage(train_df, test_df) == []


@pytest.mark.parametrize("seed", range(10))
def test_disjoint_split_never_flagged_on_small_datasets(seed):
    # small files exercise the widened uniqueness ceiling (see keys.py) --
    # confirm that widening still doesn't produce false positives on data
    # with a genuinely disjoint entity pool
    train_df, test_df = make_disjoint_split(
        n_train=80, n_test=40, n_entities_train=30, n_entities_test=30, seed=seed,
    )
    assert check_cross_split_leakage(train_df, test_df) == []


@pytest.mark.parametrize("seed", range(10))
def test_disjoint_split_never_flagged_with_multiple_candidate_columns(seed):
    # two independent disjoint entity columns on the same frame -- makes
    # sure a false positive on either candidate column would surface
    train_a, test_a = make_disjoint_split(
        n_train=600, n_test=300, n_entities_train=150, n_entities_test=150,
        entity_col="entity_a", seed=seed,
    )
    train_b, test_b = make_disjoint_split(
        n_train=600, n_test=300, n_entities_train=40, n_entities_test=40,
        entity_col="entity_b", seed=seed + 1000,
    )
    train_df = train_a.assign(entity_b=train_b["entity_b"])
    test_df = test_a.assign(entity_b=test_b["entity_b"])
    assert check_cross_split_leakage(train_df, test_df) == []


def test_identical_column_values_but_no_test_df_column_never_flagged():
    # candidate exists in train but the column simply isn't present in test
    # -- must never be treated as 100% overlap
    train_df, _ = make_disjoint_split(seed=0)
    test_df = pd.DataFrame({"unrelated": [1, 2, 3]})
    assert check_cross_split_leakage(train_df, test_df) == []
