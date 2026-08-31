from dataclasses import dataclass
from typing import List

import pandas as pd

from data_sweep.entity_leakage.findings import EntityLeakageFinding
from data_sweep.entity_leakage.keys import CandidateKey, score_candidate_keys

DEFAULT_OVERLAP_THRESHOLD = 0.02  # >2% overlap on a supposedly disjoint split is suspicious

# Severity buckets for fix-code generation (which candidate key to fix first
# when several are flagged) and for the plain-language "high/medium/low"
# label. These are independent of DEFAULT_OVERLAP_THRESHOLD, which only
# decides whether something is a finding at all.
HIGH_OVERLAP_RATIO = 0.20
MEDIUM_OVERLAP_RATIO = 0.05


@dataclass
class LeakageFinding:
    column: str
    overlap_ratio: float
    overlap_count: int
    test_entity_count: int
    candidate_key: CandidateKey
    example_overlapping_values: List[str]


def check_cross_split_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> List[LeakageFinding]:
    """Check every candidate key inferred from train_df for cross-split overlap with test_df.

    Candidacy is decided from train_df alone (the larger/reference file);
    test_df's own cardinality shape doesn't have to independently qualify —
    a smaller test split can easily fall outside the grouping band on
    sampling noise alone even when the same real entity column is present.
    """
    findings = []

    for candidate in score_candidate_keys(train_df):
        col = candidate.column
        if col not in test_df.columns:
            continue

        train_values = set(train_df[col].dropna())
        test_values = set(test_df[col].dropna())
        if len(test_values) == 0:
            continue

        overlap_values = test_values & train_values
        overlap_ratio = len(overlap_values) / len(test_values)

        if overlap_ratio > overlap_threshold:
            findings.append(LeakageFinding(
                column=col,
                overlap_ratio=overlap_ratio,
                overlap_count=len(overlap_values),
                test_entity_count=len(test_values),
                candidate_key=candidate,
                example_overlapping_values=sorted(str(v) for v in overlap_values)[:3],
            ))

    return rank_by_severity(findings)


def _severity_key(finding: LeakageFinding) -> tuple:
    return (finding.overlap_ratio, finding.overlap_count, finding.candidate_key.score)


def rank_by_severity(findings: List[LeakageFinding]) -> List[LeakageFinding]:
    """Sort leakage findings most-severe first.

    A higher overlap ratio wins; ties broken by overlap count (more affected
    entities is worse even at the same rate), then by how strong the
    underlying candidate-key evidence was.
    """
    return sorted(findings, key=_severity_key, reverse=True)


def _severity_label(overlap_ratio: float) -> str:
    if overlap_ratio >= HIGH_OVERLAP_RATIO:
        return "high"
    if overlap_ratio >= MEDIUM_OVERLAP_RATIO:
        return "medium"
    return "low"


def to_entity_leakage_findings(findings: List[LeakageFinding]) -> List[EntityLeakageFinding]:
    """Flatten LeakageFinding objects into the mode-aware EntityLeakageFinding
    shape that fix-code generation (and any other downstream consumer) uses.

    Pure reshaping -- doesn't change which columns get flagged or how
    they're ranked, so check_cross_split_leakage()'s own tests still cover
    the actual detection logic untouched.
    """
    return [
        EntityLeakageFinding(
            candidate_key=f.column,
            uniqueness_ratio=f.candidate_key.uniqueness_ratio,
            overlap_pct=f.overlap_ratio * 100,
            severity=_severity_label(f.overlap_ratio),
            mode="two_file",
            example_values=list(f.example_overlapping_values),
        )
        for f in findings
    ]
