"""Regression check: MCP tool output must match the CLI's output for the
same inputs. Both call the exact same underlying functions (leakage.py,
temporal.py, keys.py) -- this test asserts that stays true rather than
re-testing the detection logic itself (that's covered elsewhere).
"""
from dataclasses import asdict

import pandas as pd

from data_sweep.entity_leakage.findings import from_candidate_keys
from data_sweep.entity_leakage.keys import score_candidate_keys
from data_sweep.entity_leakage.leakage import check_cross_split_leakage, to_entity_leakage_findings
from data_sweep.entity_leakage.temporal import check_temporal_leakage
from data_sweep.mcp_server import audit_dataset_impl
from tests.synthetic import make_leaky_split, make_temporal_leak_dataset


def _write_csv(df, path) -> str:
    df.to_csv(path, index=False)
    return str(path)


def test_audit_dataset_two_file_matches_cli_entity_findings(tmp_path):
    train_df, test_df = make_leaky_split()
    train_path = _write_csv(train_df, tmp_path / "train.csv")
    test_path = _write_csv(test_df, tmp_path / "test.csv")

    cli_findings = to_entity_leakage_findings(check_cross_split_leakage(train_df, test_df))
    cli_dicts = [asdict(f) for f in cli_findings]

    mcp_result = audit_dataset_impl(train_path, test_path=test_path)
    mcp_dicts = [{k: v for k, v in f.items() if k != "check"} for f in mcp_result["findings"]]

    assert mcp_dicts == cli_dicts


def test_audit_dataset_single_file_matches_cli_candidate_findings(tmp_path):
    train_df, _ = make_leaky_split()
    train_path = _write_csv(train_df, tmp_path / "train.csv")

    cli_findings = from_candidate_keys(score_candidate_keys(train_df))
    cli_dicts = [asdict(f) for f in cli_findings]

    mcp_result = audit_dataset_impl(train_path)
    mcp_dicts = [{k: v for k, v in f.items() if k != "check"} for f in mcp_result["findings"]]

    assert mcp_dicts == cli_dicts


def test_audit_dataset_temporal_matches_cli_temporal_findings(tmp_path):
    df = make_temporal_leak_dataset()
    path = _write_csv(df, tmp_path / "data.csv")

    # CLI (via run_audit) also loads from the CSV rather than the in-memory
    # frame -- read it back the same way here so a CSV round-trip's float
    # precision noise doesn't look like a real MCP-vs-CLI discrepancy.
    csv_df = pd.read_csv(path)
    cli_findings = check_temporal_leakage(csv_df, "target", "cancel_date", "snapshot_date")
    cli_dicts = [asdict(f) for f in cli_findings]

    mcp_result = audit_dataset_impl(path, target="target", event_time="cancel_date", record_time="snapshot_date")
    temporal_dicts = [
        {k: v for k, v in f.items() if k != "check"}
        for f in mcp_result["findings"]
        if f["check"] == "temporal_leakage"
    ]

    assert temporal_dicts == cli_dicts
