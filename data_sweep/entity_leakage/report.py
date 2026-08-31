from typing import List

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
