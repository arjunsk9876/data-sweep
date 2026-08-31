import numpy as np
import pandas as pd
import pytest

from data_sweep.entity_leakage.baseline import compute_predictiveness


def test_perfectly_separating_feature_gives_auc_near_one():
    target = pd.Series([0] * 50 + [1] * 50)
    feature = pd.Series(list(range(50)) + list(range(100, 150)))  # class 1 always higher
    result = compute_predictiveness(feature, target)
    assert result is not None
    assert result.metric == "auc"
    assert result.score == pytest.approx(1.0)


def test_inversely_separating_feature_still_gives_high_score():
    # class 1 always lower -- raw AUC would be ~0, but discriminative power
    # is identical to the perfectly-separating case, so score should match
    target = pd.Series([0] * 50 + [1] * 50)
    feature = pd.Series(list(range(100, 150)) + list(range(50)))
    result = compute_predictiveness(feature, target)
    assert result is not None
    assert result.score == pytest.approx(1.0)


def test_random_noise_feature_gives_auc_near_half():
    rng = np.random.RandomState(0)
    target = pd.Series(rng.randint(0, 2, 2000))
    feature = pd.Series(rng.normal(0, 1, 2000))
    result = compute_predictiveness(feature, target)
    assert result is not None
    assert result.metric == "auc"
    assert 0.45 < result.score < 0.55


def test_constant_feature_returns_none():
    target = pd.Series([0] * 20 + [1] * 20)
    feature = pd.Series([5.0] * 40)
    assert compute_predictiveness(feature, target) is None


def test_too_few_rows_returns_none():
    target = pd.Series([0, 1, 0, 1, 0])
    feature = pd.Series([1, 2, 3, 4, 5])
    assert compute_predictiveness(feature, target) is None


def test_non_binary_target_returns_none_for_now():
    # continuous-target (R^2) support lands in a later commit
    target = pd.Series(np.linspace(0, 100, 50))
    feature = pd.Series(np.linspace(0, 1, 50))
    assert compute_predictiveness(feature, target) is None


def test_rows_with_missing_values_are_dropped_before_scoring():
    target = pd.Series([0] * 10 + [1] * 10 + [None] * 5)
    feature = pd.Series(list(range(20)) + [None] * 5)
    result = compute_predictiveness(feature, target)
    assert result is not None  # the 5 rows with missing values shouldn't crash or dominate


def test_non_numeric_binary_labels_work():
    target = pd.Series(["no"] * 50 + ["yes"] * 50)
    feature = pd.Series(list(range(50)) + list(range(100, 150)))
    result = compute_predictiveness(feature, target)
    assert result is not None
    assert result.score == pytest.approx(1.0)
