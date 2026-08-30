from dataclasses import dataclass, field
from typing import List

import pandas as pd

from data_sweep.entity_leakage.format_signal import detect_format_signal
from data_sweep.entity_leakage.name_signal import detect_name_signal

DEFAULT_MIN_UNIQUENESS_RATIO = 0.02
DEFAULT_MAX_UNIQUENESS_RATIO = 0.95

# score contribution of each corroborating signal; uniqueness is the gate
# (must be in the grouping band to be a candidate at all), format/name are
# additive boosts used to rank among candidates, never gates themselves.
# Format outranks name since it's evidence from the data itself; a name is
# just a label, and a renamed/anonymized column carries none at all.
UNIQUENESS_SCORE = 1.0
FORMAT_SIGNAL_BOOST = 0.15
NAME_SIGNAL_BOOST = 0.1


@dataclass
class CandidateKey:
    column: str
    uniqueness_ratio: float
    score: float
    signals: List[str] = field(default_factory=list)


def score_candidate_keys(
    df: pd.DataFrame,
    min_uniqueness_ratio: float = DEFAULT_MIN_UNIQUENESS_RATIO,
    max_uniqueness_ratio: float = DEFAULT_MAX_UNIQUENESS_RATIO,
) -> List[CandidateKey]:
    """Score every column as a candidate entity/group key.

    A column qualifies only if its uniqueness ratio (distinct non-null
    values / total rows) falls in the "grouping band" — high enough to
    suggest a real entity, low enough to rule out both a plain row id
    (~100% unique) and a low-cardinality categorical.
    """
    n_rows = len(df)
    if n_rows == 0:
        return []

    candidates = []
    for col in df.columns:
        uniqueness_ratio = df[col].nunique(dropna=True) / n_rows
        if not (min_uniqueness_ratio <= uniqueness_ratio <= max_uniqueness_ratio):
            continue

        score = UNIQUENESS_SCORE
        signals = ["uniqueness"]

        if detect_format_signal(df[col]):
            score += FORMAT_SIGNAL_BOOST
            signals.append("format")

        if detect_name_signal(col):
            score += NAME_SIGNAL_BOOST
            signals.append("name")

        candidates.append(CandidateKey(
            column=col,
            uniqueness_ratio=uniqueness_ratio,
            score=score,
            signals=signals,
        ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates
