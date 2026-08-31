from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from data_sweep.entity_leakage.format_signal import detect_format_signal
from data_sweep.entity_leakage.name_signal import detect_name_signal

DEFAULT_MIN_UNIQUENESS_RATIO = 0.02
DEFAULT_MAX_UNIQUENESS_RATIO = 0.95

# Below this row count, uniqueness_ratio is a noisy estimate: a true entity
# key can easily land above the normal 0.95 ceiling just because a small
# sample didn't happen to repeat many values. Widen the ceiling rather than
# the floor -- the floor almost never binds on small data anyway (few rows
# means even a single duplicate pushes the ratio well above 0.02).
SMALL_DATASET_ROW_THRESHOLD = 200
SMALL_DATASET_MAX_UNIQUENESS_RATIO = 0.99

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
    max_uniqueness_ratio: Optional[float] = None,
) -> List[CandidateKey]:
    """Score every column as a candidate entity/group key.

    A column qualifies only if its uniqueness ratio (distinct non-null
    values / total rows) falls in the "grouping band" — high enough to
    suggest a real entity, low enough to rule out both a plain row id
    (~100% unique) and a low-cardinality categorical.

    max_uniqueness_ratio defaults to DEFAULT_MAX_UNIQUENESS_RATIO, but on a
    small dataset (fewer than SMALL_DATASET_ROW_THRESHOLD rows) the ceiling
    is widened to SMALL_DATASET_MAX_UNIQUENESS_RATIO instead, since a small
    sample can push a true entity key's ratio above the normal ceiling on
    noise alone. Pass max_uniqueness_ratio explicitly to opt out of that
    widening and pin an exact ceiling regardless of row count.
    """
    n_rows = len(df)
    if n_rows == 0:
        return []

    if max_uniqueness_ratio is not None:
        effective_max_ratio = max_uniqueness_ratio
    elif n_rows < SMALL_DATASET_ROW_THRESHOLD:
        effective_max_ratio = SMALL_DATASET_MAX_UNIQUENESS_RATIO
    else:
        effective_max_ratio = DEFAULT_MAX_UNIQUENESS_RATIO

    candidates = []
    for col in df.columns:
        uniqueness_ratio = df[col].nunique(dropna=True) / n_rows
        if not (min_uniqueness_ratio <= uniqueness_ratio <= effective_max_ratio):
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
