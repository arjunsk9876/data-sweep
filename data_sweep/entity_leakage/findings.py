from dataclasses import dataclass
from typing import List, Optional

from data_sweep.entity_leakage.keys import CandidateKey

# Severity buckets for single-file mode, derived from candidate-key score
# rather than measured overlap (there's no test file to measure overlap
# against). A candidate with both the format and name signals firing is a
# stronger "this is really an entity key" signal than uniqueness alone.
HIGH_SCORE_THRESHOLD = 1.20  # both format (+0.15) and name (+0.1) signals
MEDIUM_SCORE_THRESHOLD = 1.05  # exactly one of the two signals


@dataclass
class EntityLeakageFinding:
    """Structured representation of one entity-leakage finding, independent
    of how it gets rendered (plain-language report, fix-code generation, or
    anything else downstream).

    overlap_pct and severity reflect actual measured cross-split overlap in
    two_file mode. In single_file mode there's no test file to measure
    overlap against, so overlap_pct is always 0.0 and severity instead
    reflects confidence that the column really is an entity/group key
    (derived from its candidate-key score) -- a leakage-risk signal for a
    future split, not an observed leak.
    """
    candidate_key: str
    uniqueness_ratio: float
    overlap_pct: float
    severity: str  # "high" / "medium" / "low"
    mode: str      # "two_file" or "single_file"
    example_values: List[str]


def _severity_from_score(score: float) -> str:
    if score >= HIGH_SCORE_THRESHOLD:
        return "high"
    if score >= MEDIUM_SCORE_THRESHOLD:
        return "medium"
    return "low"


def from_candidate_keys(candidates: List[CandidateKey]) -> List[EntityLeakageFinding]:
    """Build single_file-mode findings from candidate-key inference alone.

    There's no test file to check for cross-split overlap, so overlap_pct
    is always 0.0 here -- these are leakage-risk signals for a future
    split, not observed leaks. See to_entity_leakage_findings() in
    leakage.py for the two_file-mode equivalent, built from actual
    overlap findings instead.
    """
    return [
        EntityLeakageFinding(
            candidate_key=c.column,
            uniqueness_ratio=c.uniqueness_ratio,
            overlap_pct=0.0,
            severity=_severity_from_score(c.score),
            mode="single_file",
            example_values=[],
        )
        for c in candidates
    ]


@dataclass
class TemporalLeakageFinding:
    """One temporal-leakage finding: a feature that looks like it may have
    been computed using data from after the label event.

    Heuristic, not proof -- always "suspicious, worth investigating," never
    a guarantee (see the temporal-leakage PRD's honesty framing). Bundles
    all three raw signals so report.py can explain exactly why a feature
    was flagged, not just assert that it was:

    - name_signal_matched: does the column name look aggregation-style
      (total_, avg_, cumulative_, etc)? Weighting only, never a gate.
    - elapsed_time_correlation: correlation between the feature and how
      much time elapsed past the label event. None when --event-time /
      --record-time weren't both provided, or when it couldn't be
      meaningfully computed -- not the same as "checked and found none".
    - predictiveness_score / predictiveness_metric: how predictive the
      feature is of the target on its own ("auc" or "r2"). None when it
      couldn't be meaningfully computed.

    reduced_confidence is True whenever elapsed_time_correlation is
    unavailable because the timestamp columns weren't given at all (as
    opposed to being given but not computable) -- report.py uses it to
    label the finding "POSSIBLE" instead of "POTENTIAL" per the PRD, so a
    weaker-evidence finding is never presented with the same confidence as
    a full one.

    event_time_col carries the actual --event-time column name through so
    report.py can name it in the recommendation ("recompute before
    'cancel_date'") rather than speaking only in the abstract. None when
    timestamps weren't provided.
    """
    feature: str
    name_signal_matched: bool
    elapsed_time_correlation: Optional[float]
    predictiveness_score: Optional[float]
    predictiveness_metric: Optional[str]  # "auc" or "r2", or None if unavailable
    severity: str  # "HIGH" / "MEDIUM" / "LOW"
    reduced_confidence: bool
    event_time_col: Optional[str] = None  # for a specific recommendation ("before 'cancel_date'")
