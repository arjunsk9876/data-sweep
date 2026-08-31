from typing import List

from data_sweep.entity_leakage.findings import EntityLeakageFinding

# {candidate_key!r} rather than a literal quoted '{candidate_key}' -- repr()
# picks the right quoting/escaping automatically, so a column name that
# happens to contain a quote character still produces syntactically valid
# Python instead of a broken f-string-style substitution.
TWO_FILE_TEMPLATE = """from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(df, groups=df[{candidate_key!r}]))

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]
"""

SINGLE_FILE_TEMPLATE = """from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(df, groups=df[{candidate_key!r}]):
    train_fold = df.iloc[train_idx]
    val_fold = df.iloc[val_idx]
    # ... fit/evaluate per fold
"""

_SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}


class NoFindingsError(Exception):
    """Raised when fix-code generation is asked to run on an empty findings list."""


def _select_primary(findings: List[EntityLeakageFinding]) -> EntityLeakageFinding:
    # highest severity wins; ties broken by overlap_pct (two_file) or
    # uniqueness_ratio (single_file, where overlap_pct is always 0.0)
    return max(findings, key=lambda f: (_SEVERITY_RANK[f.severity], f.overlap_pct, f.uniqueness_ratio))


def _alt_note(finding: EntityLeakageFinding) -> str:
    if finding.mode == "two_file":
        return (
            f"# Note: '{finding.candidate_key}' also showed {finding.overlap_pct:.1f}% "
            f"overlap -- consider whether it should be the grouping key instead"
        )
    return (
        f"# Note: '{finding.candidate_key}' is also a candidate entity/group key "
        f"(uniqueness ratio {finding.uniqueness_ratio:.3f}) -- consider whether it "
        f"should be the grouping key instead"
    )


def generate_fix_code(findings: List[EntityLeakageFinding]) -> str:
    """Render a runnable Python fix for the given findings.

    Picks the highest-severity finding to fix (ties broken by overlap_pct,
    then uniqueness_ratio) and renders the template matching its mode. If
    more than one finding was passed, every other candidate gets a one-line
    comment above the code noting it as an alternative worth considering --
    never silently dropped.
    """
    if not findings:
        raise NoFindingsError("no findings to generate a fix for")

    primary = _select_primary(findings)
    template = TWO_FILE_TEMPLATE if primary.mode == "two_file" else SINGLE_FILE_TEMPLATE
    code = template.format(candidate_key=primary.candidate_key)

    others = [f for f in findings if f is not primary]
    if others:
        notes = "\n".join(_alt_note(o) for o in others)
        code = f"{notes}\n{code}"

    return code
