from dataclasses import dataclass
from typing import List

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
