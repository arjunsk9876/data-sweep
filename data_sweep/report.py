from data_sweep.findings import Finding

ISSUE_TYPE_LABELS = {
    "duplicate_rows": "Duplicate Rows",
    "missing_values": "Missing Values",
    "constant_column": "Constant Column",
}


def report(findings: list[Finding], before_shape: tuple[int, int], after_shape: tuple[int, int]) -> str:
    lines = [
        "# data-sweep report",
        "",
        "## Summary",
        f"- Rows: {before_shape[0]} -> {after_shape[0]}",
        f"- Columns: {before_shape[1]} -> {after_shape[1]}",
        f"- Issues found: {len(findings)}",
        "",
    ]

    if not findings:
        lines.append("No issues found. Data was already clean.")
        return "\n".join(lines)

    lines.append("## Findings")
    lines.append("")
    for i, f in enumerate(findings, start=1):
        label = ISSUE_TYPE_LABELS.get(f.issue_type, f.issue_type.replace("_", " ").title())
        lines.append(f"### {i}. {f.column} — {label}")
        lines.append(f"- **Confidence:** {f.confidence:.0%}")
        lines.append(f"- **Detail:** {f.detail}")
        lines.append(f"- **Action taken:** {f.action_taken}")
        lines.append("")

    return "\n".join(lines)
