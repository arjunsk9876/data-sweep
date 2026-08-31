from typing import List

from data_sweep.entity_leakage.findings import TemporalLeakageFinding
from data_sweep.entity_leakage.keys import CandidateKey
from data_sweep.entity_leakage.leakage import LeakageFinding


def format_finding(finding: LeakageFinding) -> str:
    """Render one leakage finding as a plain-language explanation with a fix."""
    ratio_pct = finding.overlap_ratio * 100
    examples = ", ".join(finding.example_overlapping_values) or "none captured"
    return (
        f"Column '{finding.column}' looks like an entity/group key, and "
        f"{finding.overlap_count} of {finding.test_entity_count} test values "
        f"({ratio_pct:.1f}%) also appear in the training data.\n"
        f"  Example overlapping values: {examples}\n"
        f"  This means rows for the same entity can land in both train and "
        f"test, so a model can partly memorize the entity instead of "
        f"learning to generalize -- test performance can look better than "
        f"it will be on genuinely unseen entities.\n"
        f"  Recommendation: split train/test by '{finding.column}' (a "
        f"group/entity split) instead of by row, so each entity appears "
        f"in only one side of the split."
    )


def format_audit_report(findings: List[LeakageFinding]) -> str:
    """Render a full audit report: a summary line plus one section per finding."""
    if not findings:
        return "No entity/group leakage detected between train and test."

    count = len(findings)
    header = f"Found {count} possible leak{'s' if count != 1 else ''} between train and test:"
    body = "\n\n".join(format_finding(f) for f in findings)
    return f"{header}\n\n{body}"


def format_single_file_report(candidates: List[CandidateKey]) -> str:
    """Render an informational report for single-file mode (no --test given).

    Without a second file there's nothing to check for cross-split
    overlap, so this only surfaces which columns look like entity/group
    keys -- useful on its own for understanding a dataset's structure,
    and as a hint for what to pass to --test once a split exists.
    """
    if not candidates:
        return (
            "No candidate entity/group key columns detected in this file.\n"
            "No --test file was provided, so this was informational only -- "
            "no leakage check was run."
        )

    count = len(candidates)
    lines = [f"Detected {count} candidate entity/group key column{'s' if count != 1 else ''}:"]
    for c in candidates:
        signals = ", ".join(c.signals)
        lines.append(f"  '{c.column}' -- uniqueness ratio {c.uniqueness_ratio:.3f} (signals: {signals})")
    lines.append("")
    lines.append(
        "No --test file was provided, so this was informational only -- no "
        "leakage check was run. Pass --test <file> to check these columns "
        "for overlap between train and test."
    )
    return "\n".join(lines)


def _temporal_signal_lines(finding: TemporalLeakageFinding) -> List[str]:
    lines = []

    if finding.name_signal_matched:
        lines.append("- Name pattern matched (aggregation-style naming, e.g. total_/avg_/cumulative_)")

    if finding.elapsed_time_correlation is not None:
        direction = (
            "value grows with time since the event -- suspicious"
            if finding.elapsed_time_correlation > 0
            else "value shrinks with time since the event"
        )
        lines.append(f"- Elapsed-time correlation: {finding.elapsed_time_correlation:.2f} ({direction})")

    if finding.predictiveness_score is not None:
        metric_label = (finding.predictiveness_metric or "score").upper()
        lines.append(
            f"- Predictiveness: single-feature {metric_label} "
            f"{finding.predictiveness_score:.2f} (unusually high for one feature)"
        )

    return lines


def format_temporal_finding(finding: TemporalLeakageFinding) -> str:
    """Render one temporal-leakage finding as a plain-language explanation.

    Heuristic, not proof -- this is always framed as "worth investigating",
    never a guarantee. A reduced_confidence finding is labeled POSSIBLE
    instead of POTENTIAL and says explicitly why it's weaker evidence
    (elapsed-time correlation couldn't be checked at all), so it's never
    mistaken for a full-confidence finding.
    """
    if finding.reduced_confidence:
        header = "POSSIBLE TEMPORAL LEAKAGE (lower confidence -- no event/record timestamps provided)"
    else:
        header = "POTENTIAL TEMPORAL LEAKAGE"

    signal_lines = _temporal_signal_lines(finding)
    signals_block = "\n".join(signal_lines) if signal_lines else "- (no additional detail available)"

    if finding.event_time_col:
        recompute_line = f"Recompute this feature using only data available before '{finding.event_time_col}', "
    else:
        recompute_line = "Recompute this feature using only data available before the label event, "

    return (
        f"{header}\n\n"
        f"Feature: {finding.feature}\n\n"
        f"Signals:\n{signals_block}\n\n"
        f"Severity: {finding.severity}\n\n"
        f"Why this matters:\n"
        f"This feature's value may include activity that happened after the "
        f"label event -- information that wouldn't be available at real "
        f"prediction time, which can make the feature look more useful than "
        f"it actually will be. This is a heuristic flag based on naming, "
        f"correlation, and predictiveness patterns, not proof of leakage --"
        f" worth investigating, not necessarily a bug.\n\n"
        f"Recommendation:\n"
        f"{recompute_line}and verify the recomputed version doesn't "
        f"correlate with elapsed time."
    )


def format_temporal_report(findings: List[TemporalLeakageFinding]) -> str:
    """Render a full temporal-leakage report: a summary line plus one
    section per finding, worst-first (callers should pass in an
    already-ranked list, e.g. from rank_temporal_findings())."""
    if not findings:
        return "No potential temporal leakage detected."

    count = len(findings)
    header = f"Found {count} feature{'s' if count != 1 else ''} with potential temporal leakage:"
    body = "\n\n".join(format_temporal_finding(f) for f in findings)
    return f"{header}\n\n{body}"
