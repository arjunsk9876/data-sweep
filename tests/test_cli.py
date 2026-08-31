import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from data_sweep.entity_leakage.cli import add_audit_subparser, run_audit
from tests.synthetic import make_disjoint_split, make_leaky_split, make_temporal_leak_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_csv(df: pd.DataFrame, path) -> str:
    df.to_csv(path, index=False)
    return str(path)


def test_add_audit_subparser_wires_expected_args():
    parser = argparse.ArgumentParser()
    add_audit_subparser(parser)
    args = parser.parse_args(["train.csv", "--test", "test.csv"])
    assert args.input_csv == "train.csv"
    assert args.test_csv == "test.csv"


def test_add_audit_subparser_test_csv_optional():
    parser = argparse.ArgumentParser()
    add_audit_subparser(parser)
    args = parser.parse_args(["train.csv"])
    assert args.input_csv == "train.csv"
    assert args.test_csv is None


def test_add_audit_subparser_fix_flag_defaults_false():
    parser = argparse.ArgumentParser()
    add_audit_subparser(parser)
    args = parser.parse_args(["train.csv"])
    assert args.fix is False


def test_add_audit_subparser_fix_flag_settable():
    parser = argparse.ArgumentParser()
    add_audit_subparser(parser)
    args = parser.parse_args(["train.csv", "--fix"])
    assert args.fix is True


def test_add_audit_subparser_fix_file_flag_defaults_none():
    parser = argparse.ArgumentParser()
    add_audit_subparser(parser)
    args = parser.parse_args(["train.csv"])
    assert args.fix_file is None


def test_add_audit_subparser_fix_file_flag_settable():
    parser = argparse.ArgumentParser()
    add_audit_subparser(parser)
    args = parser.parse_args(["train.csv", "--fix-file", "fix.py"])
    assert args.fix_file == "fix.py"


def test_add_audit_subparser_target_flag_defaults_none():
    parser = argparse.ArgumentParser()
    add_audit_subparser(parser)
    args = parser.parse_args(["train.csv"])
    assert args.target is None


def test_add_audit_subparser_target_flag_settable():
    parser = argparse.ArgumentParser()
    add_audit_subparser(parser)
    args = parser.parse_args(["train.csv", "--target", "churn"])
    assert args.target == "churn"


def test_add_audit_subparser_event_and_record_time_flags_default_none():
    parser = argparse.ArgumentParser()
    add_audit_subparser(parser)
    args = parser.parse_args(["train.csv"])
    assert args.event_time_col is None
    assert args.record_time_col is None


def test_add_audit_subparser_event_and_record_time_flags_settable():
    parser = argparse.ArgumentParser()
    add_audit_subparser(parser)
    args = parser.parse_args([
        "train.csv", "--target", "churn",
        "--event-time", "cancel_date", "--record-time", "snapshot_date",
    ])
    assert args.event_time_col == "cancel_date"
    assert args.record_time_col == "snapshot_date"


def test_add_audit_subparser_sets_description_and_examples():
    parser = argparse.ArgumentParser()
    add_audit_subparser(parser)
    assert "inferred automatically" in parser.description
    assert "--test test.csv" in parser.epilog


def test_top_level_help_runs_cleanly():
    result = subprocess.run(
        [sys.executable, "run.py", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "clean" in result.stdout
    assert "audit" in result.stdout
    assert "Traceback" not in result.stdout and "Traceback" not in result.stderr


def test_audit_help_runs_cleanly():
    result = subprocess.run(
        [sys.executable, "run.py", "audit", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--test" in result.stdout
    assert "inferred automatically" in result.stdout
    assert "Traceback" not in result.stdout and "Traceback" not in result.stderr


def test_audit_missing_file_via_console_exits_cleanly_no_traceback():
    result = subprocess.run(
        [sys.executable, "run.py", "audit", "does_not_exist.csv"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "Error:" in result.stderr
    assert "Traceback" not in result.stdout and "Traceback" not in result.stderr


def test_run_audit_single_file_mode_reports_candidates(tmp_path, capsys):
    train_df, _ = make_leaky_split(seed=0)
    train_path = _write_csv(train_df, tmp_path / "train.csv")

    run_audit(argparse.Namespace(input_csv=train_path, test_csv=None, fix=False, fix_file=None, target=None, event_time_col=None, record_time_col=None))

    out = capsys.readouterr().out
    assert "entity_id" in out
    assert "informational only" in out


def test_run_audit_two_file_mode_reports_leakage(tmp_path, capsys):
    train_df, test_df = make_leaky_split(overlap_entities=20, seed=0)
    train_path = _write_csv(train_df, tmp_path / "train.csv")
    test_path = _write_csv(test_df, tmp_path / "test.csv")

    run_audit(argparse.Namespace(input_csv=train_path, test_csv=test_path, fix=False, fix_file=None, target=None, event_time_col=None, record_time_col=None))

    out = capsys.readouterr().out
    assert "entity_id" in out
    assert "Found 1 possible leak" in out


def test_run_audit_two_file_mode_no_leakage(tmp_path, capsys):
    train_df, test_df = make_disjoint_split(seed=0)
    train_path = _write_csv(train_df, tmp_path / "train.csv")
    test_path = _write_csv(test_df, tmp_path / "test.csv")

    run_audit(argparse.Namespace(input_csv=train_path, test_csv=test_path, fix=False, fix_file=None, target=None, event_time_col=None, record_time_col=None))

    out = capsys.readouterr().out
    assert "No entity/group leakage detected" in out


def test_run_audit_missing_train_file_exits_cleanly(tmp_path, capsys):
    missing_path = str(tmp_path / "does_not_exist.csv")

    with pytest.raises(SystemExit) as excinfo:
        run_audit(argparse.Namespace(input_csv=missing_path, test_csv=None, fix=False, fix_file=None, target=None, event_time_col=None, record_time_col=None))

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "Error:" in err
    assert "does_not_exist.csv" in err


def test_run_audit_fix_two_file_mode_prints_fix_code(tmp_path, capsys):
    train_df, test_df = make_leaky_split(overlap_entities=20, seed=0)
    train_path = _write_csv(train_df, tmp_path / "train.csv")
    test_path = _write_csv(test_df, tmp_path / "test.csv")

    run_audit(argparse.Namespace(input_csv=train_path, test_csv=test_path, fix=True, fix_file=None, target=None, event_time_col=None, record_time_col=None))

    out = capsys.readouterr().out
    assert "GroupShuffleSplit" in out
    assert "entity_id" in out


def test_run_audit_fix_two_file_mode_no_leakage_says_nothing_to_fix(tmp_path, capsys):
    train_df, test_df = make_disjoint_split(seed=0)
    train_path = _write_csv(train_df, tmp_path / "train.csv")
    test_path = _write_csv(test_df, tmp_path / "test.csv")

    run_audit(argparse.Namespace(input_csv=train_path, test_csv=test_path, fix=True, fix_file=None, target=None, event_time_col=None, record_time_col=None))

    out = capsys.readouterr().out
    assert "Nothing to fix" in out
    assert "GroupShuffleSplit" not in out


def test_run_audit_fix_single_file_mode_prints_fix_code(tmp_path, capsys):
    train_df, _ = make_leaky_split(seed=0)
    train_path = _write_csv(train_df, tmp_path / "train.csv")

    run_audit(argparse.Namespace(input_csv=train_path, test_csv=None, fix=True, fix_file=None, target=None, event_time_col=None, record_time_col=None))

    out = capsys.readouterr().out
    assert "GroupKFold" in out
    assert "entity_id" in out


def test_run_audit_no_fix_flag_does_not_print_fix_code(tmp_path, capsys):
    train_df, test_df = make_leaky_split(overlap_entities=20, seed=0)
    train_path = _write_csv(train_df, tmp_path / "train.csv")
    test_path = _write_csv(test_df, tmp_path / "test.csv")

    run_audit(argparse.Namespace(input_csv=train_path, test_csv=test_path, fix=False, fix_file=None, target=None, event_time_col=None, record_time_col=None))

    out = capsys.readouterr().out
    assert "GroupShuffleSplit" not in out


def test_run_audit_fix_sklearn_missing_exits_cleanly(tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn", None)
    train_df, test_df = make_leaky_split(overlap_entities=20, seed=0)
    train_path = _write_csv(train_df, tmp_path / "train.csv")
    test_path = _write_csv(test_df, tmp_path / "test.csv")

    with pytest.raises(SystemExit) as excinfo:
        run_audit(argparse.Namespace(input_csv=train_path, test_csv=test_path, fix=True, fix_file=None, target=None, event_time_col=None, record_time_col=None))

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "Error:" in err
    assert "pip install scikit-learn" in err


def test_run_audit_fix_file_writes_fix_code(tmp_path, capsys):
    train_df, test_df = make_leaky_split(overlap_entities=20, seed=0)
    train_path = _write_csv(train_df, tmp_path / "train.csv")
    test_path = _write_csv(test_df, tmp_path / "test.csv")
    fix_path = tmp_path / "fix.py"

    run_audit(argparse.Namespace(input_csv=train_path, test_csv=test_path, fix=False, fix_file=str(fix_path), target=None, event_time_col=None, record_time_col=None))

    assert fix_path.exists()
    content = fix_path.read_text()
    assert "GroupShuffleSplit" in content
    assert "entity_id" in content

    out = capsys.readouterr().out
    assert f"Wrote fix code to {fix_path}" in out
    assert "GroupShuffleSplit" not in out  # not also dumped to stdout without --fix


def test_run_audit_fix_and_fix_file_together(tmp_path, capsys):
    train_df, test_df = make_leaky_split(overlap_entities=20, seed=0)
    train_path = _write_csv(train_df, tmp_path / "train.csv")
    test_path = _write_csv(test_df, tmp_path / "test.csv")
    fix_path = tmp_path / "fix.py"

    run_audit(argparse.Namespace(input_csv=train_path, test_csv=test_path, fix=True, fix_file=str(fix_path), target=None, event_time_col=None, record_time_col=None))

    assert fix_path.exists()
    out = capsys.readouterr().out
    assert "GroupShuffleSplit" in out  # printed
    assert f"Wrote fix code to {fix_path}" in out  # and written


def test_run_audit_fix_file_no_leakage_writes_nothing(tmp_path, capsys):
    train_df, test_df = make_disjoint_split(seed=0)
    train_path = _write_csv(train_df, tmp_path / "train.csv")
    test_path = _write_csv(test_df, tmp_path / "test.csv")
    fix_path = tmp_path / "fix.py"

    run_audit(argparse.Namespace(input_csv=train_path, test_csv=test_path, fix=False, fix_file=str(fix_path), target=None, event_time_col=None, record_time_col=None))

    assert not fix_path.exists()
    out = capsys.readouterr().out
    assert "Nothing to fix" in out


def _temporal_args(train_path, **overrides):
    defaults = dict(
        input_csv=train_path, test_csv=None, fix=False, fix_file=None,
        target=None, event_time_col=None, record_time_col=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_run_audit_temporal_check_full_confidence_flags_leaked_feature(tmp_path, capsys):
    df = make_temporal_leak_dataset(seed=0)
    train_path = _write_csv(df, tmp_path / "train.csv")

    run_audit(_temporal_args(
        train_path, target="target", event_time_col="cancel_date", record_time_col="snapshot_date",
    ))

    out = capsys.readouterr().out
    assert "POTENTIAL TEMPORAL LEAKAGE" in out
    assert "total_purchases" in out
    assert "Severity: HIGH" in out


def test_run_audit_temporal_check_reduced_confidence_without_timestamps(tmp_path, capsys):
    df = make_temporal_leak_dataset(seed=0)
    train_path = _write_csv(df, tmp_path / "train.csv")

    run_audit(_temporal_args(train_path, target="target"))

    out = capsys.readouterr().out
    assert "lower confidence" in out
    assert "no event/record timestamps provided" in out


def test_run_audit_no_target_means_no_temporal_check(tmp_path, capsys):
    df = make_temporal_leak_dataset(seed=0)
    train_path = _write_csv(df, tmp_path / "train.csv")

    run_audit(_temporal_args(train_path))

    out = capsys.readouterr().out
    assert "TEMPORAL LEAKAGE" not in out


def test_run_audit_unknown_target_column_exits_cleanly(tmp_path, capsys):
    df = make_temporal_leak_dataset(seed=0)
    train_path = _write_csv(df, tmp_path / "train.csv")

    with pytest.raises(SystemExit) as excinfo:
        run_audit(_temporal_args(train_path, target="does_not_exist"))

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "Error:" in err
    assert "--target" in err
    assert "does_not_exist" in err


def test_run_audit_unknown_event_time_column_exits_cleanly(tmp_path, capsys):
    df = make_temporal_leak_dataset(seed=0)
    train_path = _write_csv(df, tmp_path / "train.csv")

    with pytest.raises(SystemExit) as excinfo:
        run_audit(_temporal_args(
            train_path, target="target", event_time_col="does_not_exist", record_time_col="snapshot_date",
        ))

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "--event-time" in err
    assert "does_not_exist" in err


def test_run_audit_unknown_record_time_column_exits_cleanly(tmp_path, capsys):
    df = make_temporal_leak_dataset(seed=0)
    train_path = _write_csv(df, tmp_path / "train.csv")

    with pytest.raises(SystemExit) as excinfo:
        run_audit(_temporal_args(
            train_path, target="target", event_time_col="cancel_date", record_time_col="does_not_exist",
        ))

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "--record-time" in err
    assert "does_not_exist" in err


def test_run_audit_malformed_timestamp_column_does_not_crash(tmp_path, capsys):
    # the column exists (so it passes the unknown-column check) but its
    # values aren't parseable as dates -- must degrade gracefully, not
    # raise a raw parsing exception
    df = make_temporal_leak_dataset(seed=0)
    df["snapshot_date"] = ["not a date"] * len(df)
    train_path = _write_csv(df, tmp_path / "train.csv")

    run_audit(_temporal_args(
        train_path, target="target", event_time_col="cancel_date", record_time_col="snapshot_date",
    ))

    out = capsys.readouterr().out
    assert "Traceback" not in out
    assert "total_purchases" in out  # still flagged via name + predictiveness


def test_run_audit_combines_entity_and_temporal_checks(tmp_path, capsys):
    # both --test and --target given -- both checks should run in one pass
    train_df, test_df = make_leaky_split(overlap_entities=20, seed=0)
    temporal_df = make_temporal_leak_dataset(seed=0, n_rows=len(train_df))
    combined_train = pd.concat([train_df.reset_index(drop=True), temporal_df.reset_index(drop=True)], axis=1)
    train_path = _write_csv(combined_train, tmp_path / "train.csv")
    test_path = _write_csv(test_df, tmp_path / "test.csv")

    run_audit(argparse.Namespace(
        input_csv=train_path, test_csv=test_path, fix=False, fix_file=None,
        target="target", event_time_col="cancel_date", record_time_col="snapshot_date",
    ))

    out = capsys.readouterr().out
    assert "Found 1 possible leak" in out  # entity leakage report
    assert "POTENTIAL TEMPORAL LEAKAGE" in out  # temporal leakage report


def test_run_audit_missing_test_file_exits_cleanly(tmp_path, capsys):
    train_df, _ = make_leaky_split(seed=0)
    train_path = _write_csv(train_df, tmp_path / "train.csv")
    missing_path = str(tmp_path / "does_not_exist.csv")

    with pytest.raises(SystemExit) as excinfo:
        run_audit(argparse.Namespace(input_csv=train_path, test_csv=missing_path, fix=False, fix_file=None, target=None, event_time_col=None, record_time_col=None))

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "Error:" in err
