from dataclasses import dataclass
from typing import List


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
