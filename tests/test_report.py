from data_sweep.entity_leakage.findings import TemporalLeakageFinding
from data_sweep.entity_leakage.keys import CandidateKey
from data_sweep.entity_leakage.leakage import LeakageFinding
from data_sweep.entity_leakage.report import (
    format_audit_report,
    format_finding,
    format_single_file_report,
    format_temporal_finding,
    format_temporal_report,
)


def _finding(column="entity_id", overlap_ratio=0.2, overlap_count=20, test_entity_count=100, examples=None):
    return LeakageFinding(
        column=column,
        overlap_ratio=overlap_ratio,
        overlap_count=overlap_count,
        test_entity_count=test_entity_count,
        candidate_key=CandidateKey(column=column, uniqueness_ratio=0.2, score=1.0, signals=["uniqueness"]),
        example_overlapping_values=examples if examples is not None else ["E1", "E2", "E3"],
    )


def test_format_finding_includes_column_name():
    text = format_finding(_finding(column="customer_id"))
    assert "customer_id" in text


def test_format_finding_includes_overlap_counts():
    text = format_finding(_finding(overlap_count=20, test_entity_count=100))
    assert "20" in text
    assert "100" in text


def test_format_finding_includes_percentage():
    text = format_finding(_finding(overlap_ratio=0.2))
    assert "20.0%" in text


def test_format_finding_includes_example_values():
    text = format_finding(_finding(examples=["E1", "E2", "E3"]))
    assert "E1, E2, E3" in text


def test_format_finding_handles_no_examples():
    text = format_finding(_finding(examples=[]))
    assert "none captured" in text


def test_format_finding_includes_recommendation():
    text = format_finding(_finding(column="household_id"))
    assert "Recommendation" in text
    assert "household_id" in text.split("Recommendation")[1]


def test_format_audit_report_no_findings():
    text = format_audit_report([])
    assert "No entity/group leakage detected" in text


def test_format_audit_report_singular_leak():
    text = format_audit_report([_finding()])
    assert "Found 1 possible leak " in text or text.startswith("Found 1 possible leak:")
    assert "leaks" not in text.split("\n")[0]


def test_format_audit_report_plural_leaks():
    findings = [_finding(column="a"), _finding(column="b")]
    text = format_audit_report(findings)
    assert "Found 2 possible leaks" in text


def test_format_audit_report_includes_every_finding():
    findings = [_finding(column="a"), _finding(column="b"), _finding(column="c")]
    text = format_audit_report(findings)
    assert "'a'" in text
    assert "'b'" in text
    assert "'c'" in text


def _candidate(column="entity_id", uniqueness_ratio=0.2, score=1.0, signals=None):
    return CandidateKey(
        column=column,
        uniqueness_ratio=uniqueness_ratio,
        score=score,
        signals=signals if signals is not None else ["uniqueness"],
    )


def test_format_single_file_report_no_candidates():
    text = format_single_file_report([])
    assert "No candidate entity/group key columns detected" in text
    assert "informational only" in text


def test_format_single_file_report_lists_column_and_ratio():
    text = format_single_file_report([_candidate(column="customer_id", uniqueness_ratio=0.234)])
    assert "customer_id" in text
    assert "0.234" in text


def test_format_single_file_report_lists_signals():
    text = format_single_file_report([_candidate(signals=["uniqueness", "format", "name"])])
    assert "uniqueness, format, name" in text


def test_format_single_file_report_singular_plural_count():
    text_one = format_single_file_report([_candidate(column="a")])
    assert "Detected 1 candidate entity/group key column:" in text_one

    text_many = format_single_file_report([_candidate(column="a"), _candidate(column="b")])
    assert "Detected 2 candidate entity/group key columns:" in text_many


def test_format_single_file_report_mentions_test_flag():
    text = format_single_file_report([_candidate()])
    assert "--test" in text


def test_format_single_file_report_includes_every_candidate():
    candidates = [_candidate(column="a"), _candidate(column="b"), _candidate(column="c")]
    text = format_single_file_report(candidates)
    assert "'a'" in text
    assert "'b'" in text
    assert "'c'" in text


def _temporal_finding(**overrides):
    defaults = dict(
        feature="total_purchases",
        name_signal_matched=True,
        elapsed_time_correlation=0.81,
        predictiveness_score=0.93,
        predictiveness_metric="auc",
        severity="HIGH",
        reduced_confidence=False,
        event_time_col="cancel_date",
    )
    defaults.update(overrides)
    return TemporalLeakageFinding(**defaults)


def test_format_temporal_finding_full_confidence_header():
    text = format_temporal_finding(_temporal_finding(reduced_confidence=False))
    assert "POTENTIAL TEMPORAL LEAKAGE" in text
    assert "POSSIBLE" not in text


def test_format_temporal_finding_reduced_confidence_header():
    text = format_temporal_finding(_temporal_finding(reduced_confidence=True, elapsed_time_correlation=None))
    assert "POSSIBLE TEMPORAL LEAKAGE" in text
    assert "lower confidence" in text
    assert "no event/record timestamps provided" in text


def test_format_temporal_finding_includes_feature_name():
    text = format_temporal_finding(_temporal_finding(feature="avg_response_time"))
    assert "avg_response_time" in text


def test_format_temporal_finding_includes_name_signal_line_only_when_matched():
    matched = format_temporal_finding(_temporal_finding(name_signal_matched=True))
    assert "Name pattern matched" in matched

    unmatched = format_temporal_finding(_temporal_finding(name_signal_matched=False))
    assert "Name pattern matched" not in unmatched


def test_format_temporal_finding_includes_elapsed_correlation_when_available():
    text = format_temporal_finding(_temporal_finding(elapsed_time_correlation=0.81))
    assert "Elapsed-time correlation: 0.81" in text
    assert "suspicious" in text


def test_format_temporal_finding_omits_elapsed_correlation_when_unavailable():
    text = format_temporal_finding(_temporal_finding(elapsed_time_correlation=None, reduced_confidence=True))
    assert "Elapsed-time correlation" not in text


def test_format_temporal_finding_includes_predictiveness_metric_and_score():
    text = format_temporal_finding(_temporal_finding(predictiveness_score=0.93, predictiveness_metric="auc"))
    assert "AUC" in text
    assert "0.93" in text


def test_format_temporal_finding_includes_severity():
    text = format_temporal_finding(_temporal_finding(severity="MEDIUM"))
    assert "Severity: MEDIUM" in text


def test_format_temporal_finding_recommendation_names_event_time_column():
    text = format_temporal_finding(_temporal_finding(event_time_col="cancel_date"))
    assert "before 'cancel_date'" in text


def test_format_temporal_finding_recommendation_falls_back_without_event_time_column():
    text = format_temporal_finding(_temporal_finding(event_time_col=None))
    assert "before the label event" in text


def test_format_temporal_finding_frames_as_heuristic_not_proof():
    text = format_temporal_finding(_temporal_finding())
    assert "not proof" in text
    assert "worth investigating" in text


def test_format_temporal_report_no_findings():
    text = format_temporal_report([])
    assert "No potential temporal leakage detected." in text


def test_format_temporal_report_singular_plural():
    text_one = format_temporal_report([_temporal_finding(feature="a")])
    assert "Found 1 feature with potential temporal leakage:" in text_one

    text_many = format_temporal_report([_temporal_finding(feature="a"), _temporal_finding(feature="b")])
    assert "Found 2 features with potential temporal leakage:" in text_many


def test_format_temporal_report_includes_every_finding():
    findings = [_temporal_finding(feature="a"), _temporal_finding(feature="b"), _temporal_finding(feature="c")]
    text = format_temporal_report(findings)
    assert "Feature: a" in text
    assert "Feature: b" in text
    assert "Feature: c" in text
