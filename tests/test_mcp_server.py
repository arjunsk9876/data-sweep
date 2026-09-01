from pathlib import Path

import pandas as pd
import pytest

from data_sweep.mcp_server import (
    MAX_FILE_SIZE_BYTES,
    MAX_ROWS,
    ToolError,
    audit_dataset_impl,
    generate_fix_impl,
    list_checks_impl,
)
from tests.synthetic import make_disjoint_split, make_leaky_split, make_temporal_leak_dataset


def _write_csv(df: pd.DataFrame, path) -> str:
    df.to_csv(path, index=False)
    return str(path)


# --- audit_dataset_impl -----------------------------------------------------

def test_audit_dataset_two_file_finds_entity_leak(tmp_path):
    train_df, test_df = make_leaky_split()
    train_path = _write_csv(train_df, tmp_path / "train.csv")
    test_path = _write_csv(test_df, tmp_path / "test.csv")

    result = audit_dataset_impl(train_path, test_path=test_path)

    assert result["path"] == train_path
    assert result["test_path"] == test_path
    assert result["finding_count"] == len(result["findings"])
    assert any(f["check"] == "entity_leakage" and f["candidate_key"] == "entity_id" for f in result["findings"])


def test_audit_dataset_two_file_no_leak_is_clean(tmp_path):
    train_df, test_df = make_disjoint_split()
    train_path = _write_csv(train_df, tmp_path / "train.csv")
    test_path = _write_csv(test_df, tmp_path / "test.csv")

    result = audit_dataset_impl(train_path, test_path=test_path)

    assert result["findings"] == []
    assert result["finding_count"] == 0


def test_audit_dataset_single_file_mode(tmp_path):
    train_df, _ = make_leaky_split()
    train_path = _write_csv(train_df, tmp_path / "train.csv")

    result = audit_dataset_impl(train_path)

    assert result["test_path"] is None
    assert all(f["mode"] == "single_file" for f in result["findings"])


def test_audit_dataset_target_runs_temporal_check(tmp_path):
    df = make_temporal_leak_dataset()
    path = _write_csv(df, tmp_path / "data.csv")

    result = audit_dataset_impl(
        path, target="target", event_time="cancel_date", record_time="snapshot_date"
    )

    checks = {f["check"] for f in result["findings"]}
    assert "temporal_leakage" in checks
    assert any(f["feature"] == "total_purchases" for f in result["findings"] if f["check"] == "temporal_leakage")


def test_audit_dataset_no_target_skips_temporal_check(tmp_path):
    df = make_temporal_leak_dataset()
    path = _write_csv(df, tmp_path / "data.csv")

    result = audit_dataset_impl(path)

    assert all(f["check"] == "entity_leakage" for f in result["findings"])


def test_audit_dataset_missing_file_raises_tool_error():
    with pytest.raises(ToolError, match="file not found"):
        audit_dataset_impl("/no/such/file.csv")


def test_audit_dataset_directory_path_raises_tool_error(tmp_path):
    with pytest.raises(ToolError, match="not a file"):
        audit_dataset_impl(str(tmp_path))


def test_audit_dataset_malformed_csv_raises_tool_error(tmp_path):
    bad_path = tmp_path / "bad.csv"
    bad_path.write_bytes(b"\x00\x01\x02not,a,csv\xff\xfe")

    with pytest.raises(ToolError):
        audit_dataset_impl(str(bad_path))


def test_audit_dataset_unknown_target_column_raises_tool_error(tmp_path):
    train_df, _ = make_leaky_split()
    path = _write_csv(train_df, tmp_path / "train.csv")

    with pytest.raises(ToolError, match="unknown target column"):
        audit_dataset_impl(path, target="does_not_exist")


def test_audit_dataset_unknown_event_time_column_raises_tool_error(tmp_path):
    df = make_temporal_leak_dataset()
    path = _write_csv(df, tmp_path / "data.csv")

    with pytest.raises(ToolError, match="unknown event_time column"):
        audit_dataset_impl(path, target="target", event_time="nope")


def test_audit_dataset_file_over_size_limit_raises_tool_error(tmp_path, monkeypatch):
    train_df, _ = make_leaky_split(n_train=50, n_entities_train=20)
    path = _write_csv(train_df, tmp_path / "train.csv")

    import data_sweep.mcp_server as mcp_server_mod
    monkeypatch.setattr(mcp_server_mod, "MAX_FILE_SIZE_BYTES", 1)

    with pytest.raises(ToolError, match="MB"):
        audit_dataset_impl(path)


def test_audit_dataset_row_count_over_limit_raises_tool_error(tmp_path, monkeypatch):
    train_df, _ = make_leaky_split(n_train=50, n_entities_train=20)
    path = _write_csv(train_df, tmp_path / "train.csv")

    import data_sweep.mcp_server as mcp_server_mod
    monkeypatch.setattr(mcp_server_mod, "MAX_ROWS", 1)

    with pytest.raises(ToolError, match="rows"):
        audit_dataset_impl(path)


def test_audit_dataset_timeout_raises_tool_error(tmp_path, monkeypatch):
    import time

    train_df, _ = make_leaky_split(n_train=50, n_entities_train=20)
    path = _write_csv(train_df, tmp_path / "train.csv")

    import data_sweep.mcp_server as mcp_server_mod

    def _slow_score_candidate_keys(df):
        time.sleep(0.5)
        return []

    monkeypatch.setattr(mcp_server_mod, "AUDIT_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(mcp_server_mod, "score_candidate_keys", _slow_score_candidate_keys)

    with pytest.raises(ToolError, match="timed out"):
        audit_dataset_impl(path)


# --- generate_fix_impl -------------------------------------------------------

def test_generate_fix_two_file_finding_produces_group_shuffle_split():
    findings = [{
        "candidate_key": "entity_id",
        "uniqueness_ratio": 0.3,
        "overlap_pct": 25.0,
        "severity": "high",
        "mode": "two_file",
        "example_values": ["E1"],
        "check": "entity_leakage",
    }]

    result = generate_fix_impl(findings)

    assert "GroupShuffleSplit" in result["code"]
    assert "entity_id" in result["code"]


def test_generate_fix_single_file_finding_produces_group_kfold():
    findings = [{
        "candidate_key": "entity_id",
        "uniqueness_ratio": 0.3,
        "overlap_pct": 0.0,
        "severity": "medium",
        "mode": "single_file",
        "example_values": [],
        "check": "entity_leakage",
    }]

    result = generate_fix_impl(findings)

    assert "GroupKFold" in result["code"]


def test_generate_fix_empty_findings_raises_tool_error():
    with pytest.raises(ToolError, match="no findings"):
        generate_fix_impl([])


def test_generate_fix_missing_required_field_raises_tool_error():
    with pytest.raises(ToolError, match="missing required field"):
        generate_fix_impl([{"severity": "high", "mode": "single_file"}])


# --- list_checks_impl ---------------------------------------------------------

def test_list_checks_lists_both_checks():
    result = list_checks_impl()
    names = {c["name"] for c in result["checks"]}
    assert names == {"entity_leakage", "temporal_leakage"}


def test_list_checks_entries_have_description_and_inputs():
    result = list_checks_impl()
    for check in result["checks"]:
        assert check["description"]
        assert check["inputs"]
