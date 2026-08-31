"""Reduced-confidence mode correctness sweep.

The PRD frames this as a UX correctness requirement, not just a
detection one: reduced-confidence findings must be clearly
distinguishable from full-confidence ones, every time, not just in one
hand-picked example.
"""
import pytest

from data_sweep.entity_leakage.report import format_temporal_finding
from data_sweep.entity_leakage.temporal import check_temporal_leakage
from tests.synthetic import make_temporal_leak_dataset


@pytest.mark.parametrize("seed", range(15))
def test_no_timestamps_every_finding_is_reduced_confidence(seed):
    df = make_temporal_leak_dataset(seed=seed)
    findings = check_temporal_leakage(df, target_col="target")
    assert len(findings) > 0  # sanity: the leaked feature should still surface something
    assert all(f.reduced_confidence is True for f in findings)
    assert all(f.elapsed_time_correlation is None for f in findings)


@pytest.mark.parametrize("seed", range(15))
def test_only_event_time_given_is_still_reduced_confidence(seed):
    df = make_temporal_leak_dataset(seed=seed)
    findings = check_temporal_leakage(df, target_col="target", event_time_col="cancel_date")
    assert all(f.reduced_confidence is True for f in findings)
    assert all(f.elapsed_time_correlation is None for f in findings)


@pytest.mark.parametrize("seed", range(15))
def test_only_record_time_given_is_still_reduced_confidence(seed):
    df = make_temporal_leak_dataset(seed=seed)
    findings = check_temporal_leakage(df, target_col="target", record_time_col="snapshot_date")
    assert all(f.reduced_confidence is True for f in findings)
    assert all(f.elapsed_time_correlation is None for f in findings)


@pytest.mark.parametrize("seed", range(15))
def test_reduced_confidence_never_reaches_high(seed):
    # HIGH needs all three signals; without elapsed-time correlation only
    # two are ever available, so HIGH must be structurally unreachable
    df = make_temporal_leak_dataset(seed=seed)
    findings = check_temporal_leakage(df, target_col="target")
    assert all(f.severity != "HIGH" for f in findings)


@pytest.mark.parametrize("seed", range(15))
def test_full_confidence_never_marked_reduced(seed):
    df = make_temporal_leak_dataset(seed=seed)
    findings = check_temporal_leakage(
        df, target_col="target", event_time_col="cancel_date", record_time_col="snapshot_date",
    )
    assert all(f.reduced_confidence is False for f in findings)


@pytest.mark.parametrize("seed", range(15))
def test_report_text_distinguishes_reduced_from_full_confidence(seed):
    df = make_temporal_leak_dataset(seed=seed)

    reduced = check_temporal_leakage(df, target_col="target")
    full = check_temporal_leakage(
        df, target_col="target", event_time_col="cancel_date", record_time_col="snapshot_date",
    )

    for f in reduced:
        text = format_temporal_finding(f)
        assert "POSSIBLE TEMPORAL LEAKAGE" in text
        assert "lower confidence" in text
        assert "POTENTIAL TEMPORAL LEAKAGE" not in text

    for f in full:
        text = format_temporal_finding(f)
        assert "POTENTIAL TEMPORAL LEAKAGE" in text
        assert "lower confidence" not in text
        assert "POSSIBLE TEMPORAL LEAKAGE" not in text
