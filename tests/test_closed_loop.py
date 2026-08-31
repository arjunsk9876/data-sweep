"""Closed-loop validation (the PRD's strongest proof a fix actually works):
generate a fix from a real leaky dataset, apply it, then re-run the same
detection the fix claims to resolve and confirm nothing is flagged anymore.
Not just "the code runs" (test_fix_execution.py) or "the resulting split is
disjoint on the fixed column" -- specifically that a second audit pass, the
same check a user would actually run, comes back clean.
"""
import pandas as pd
import pytest

from data_sweep.entity_leakage.fixes import generate_fix_code
from data_sweep.entity_leakage.leakage import check_cross_split_leakage, to_entity_leakage_findings
from tests.synthetic import make_leaky_split


def _apply_two_file_fix(train_df: pd.DataFrame, test_df: pd.DataFrame):
    findings = check_cross_split_leakage(train_df, test_df)
    entity_findings = to_entity_leakage_findings(findings)
    code = generate_fix_code(entity_findings)

    df = pd.concat([train_df, test_df], ignore_index=True)
    namespace = {"df": df}
    exec(code, namespace)
    return namespace["train_df"], namespace["test_df"]


def test_closed_loop_resolves_leakage():
    train_df, test_df = make_leaky_split(overlap_entities=20, seed=0)

    findings_before = check_cross_split_leakage(train_df, test_df)
    assert len(findings_before) == 1  # leak confirmed present before the fix

    fixed_train, fixed_test = _apply_two_file_fix(train_df, test_df)

    findings_after = check_cross_split_leakage(fixed_train, fixed_test)
    assert findings_after == []  # leak confirmed resolved after the fix


@pytest.mark.parametrize("seed", range(10))
def test_closed_loop_resolves_leakage_across_seeds(seed):
    train_df, test_df = make_leaky_split(overlap_entities=20, seed=seed)

    findings_before = check_cross_split_leakage(train_df, test_df)
    assert len(findings_before) == 1

    fixed_train, fixed_test = _apply_two_file_fix(train_df, test_df)

    findings_after = check_cross_split_leakage(fixed_train, fixed_test)
    assert findings_after == []


def test_closed_loop_resolves_leakage_at_larger_scale():
    train_df, test_df = make_leaky_split(
        n_train=5000, n_test=2000,
        n_entities_train=800, n_entities_test_only=800,
        overlap_entities=150, seed=1,
    )

    findings_before = check_cross_split_leakage(train_df, test_df)
    assert len(findings_before) == 1

    fixed_train, fixed_test = _apply_two_file_fix(train_df, test_df)

    findings_after = check_cross_split_leakage(fixed_train, fixed_test)
    assert findings_after == []


def test_closed_loop_preserves_all_rows():
    # the fix must not silently drop data -- every row from both original
    # files should land in exactly one of the two resulting splits
    train_df, test_df = make_leaky_split(overlap_entities=20, seed=2)
    total_rows_before = len(train_df) + len(test_df)

    fixed_train, fixed_test = _apply_two_file_fix(train_df, test_df)

    assert len(fixed_train) + len(fixed_test) == total_rows_before
