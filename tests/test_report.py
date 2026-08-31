from data_sweep.entity_leakage.keys import CandidateKey
from data_sweep.entity_leakage.leakage import LeakageFinding
from data_sweep.entity_leakage.report import format_audit_report, format_finding, format_single_file_report


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
