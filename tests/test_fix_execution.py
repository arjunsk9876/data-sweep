"""Confirms generated fix code isn't just syntactically valid (test_fixes.py
already checks that with compile()) but actually runs, unmodified, against
real data shaped like the synthetic leak datasets used elsewhere -- and does
what it claims: produces a group split with no entity crossing sides.
"""
import pandas as pd

from data_sweep.entity_leakage.findings import from_candidate_keys
from data_sweep.entity_leakage.fixes import generate_fix_code
from data_sweep.entity_leakage.keys import score_candidate_keys
from data_sweep.entity_leakage.leakage import check_cross_split_leakage, to_entity_leakage_findings
from tests.synthetic import make_leaky_split


def test_two_file_fix_code_executes_and_produces_disjoint_split():
    train_df, test_df = make_leaky_split(overlap_entities=20, seed=0)
    findings = check_cross_split_leakage(train_df, test_df)
    entity_findings = to_entity_leakage_findings(findings)
    code = generate_fix_code(entity_findings)

    # the fix re-splits a combined dataframe -- that's the "existing split"
    # being fixed, per the PRD's two_file template
    df = pd.concat([train_df, test_df], ignore_index=True)
    namespace = {"df": df}

    exec(code, namespace)  # must not raise

    assert "train_df" in namespace and "test_df" in namespace
    fixed_train, fixed_test = namespace["train_df"], namespace["test_df"]
    assert len(fixed_train) + len(fixed_test) == len(df)
    assert len(fixed_train) > 0 and len(fixed_test) > 0

    entity_col = entity_findings[0].candidate_key
    train_entities = set(fixed_train[entity_col])
    test_entities = set(fixed_test[entity_col])
    assert train_entities.isdisjoint(test_entities)


def test_single_file_fix_code_executes_and_produces_disjoint_folds():
    train_df, _ = make_leaky_split(seed=0)
    candidates = score_candidate_keys(train_df)
    entity_findings = from_candidate_keys(candidates)
    code = generate_fix_code(entity_findings)

    namespace = {"df": train_df}
    exec(code, namespace)  # must not raise, loop must run to completion

    assert "train_fold" in namespace and "val_fold" in namespace
    assert len(namespace["train_fold"]) > 0
    assert len(namespace["val_fold"]) > 0

    entity_col = entity_findings[0].candidate_key
    train_entities = set(namespace["train_fold"][entity_col])
    val_entities = set(namespace["val_fold"][entity_col])
    assert train_entities.isdisjoint(val_entities)


def test_two_file_fix_code_with_multiple_candidates_still_executes():
    # alt-candidate comment lines shouldn't break execution of the real code
    train_df, test_df = make_leaky_split(overlap_entities=20, seed=0)
    train_df["session_id"] = [f"S{i % 60}" for i in range(len(train_df))]
    test_df["session_id"] = [f"S{i % 60}" for i in range(len(test_df))]

    findings = check_cross_split_leakage(train_df, test_df)
    entity_findings = to_entity_leakage_findings(findings)
    assert len(entity_findings) >= 2  # both entity_id and session_id should be candidates

    code = generate_fix_code(entity_findings)
    assert code.count("# Note:") >= 1

    df = pd.concat([train_df, test_df], ignore_index=True)
    namespace = {"df": df}
    exec(code, namespace)

    assert "train_df" in namespace and "test_df" in namespace
