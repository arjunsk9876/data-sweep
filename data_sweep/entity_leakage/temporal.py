import re

from typing import List

DEFAULT_AGGREGATION_KEYWORDS = [
    "total", "avg", "cumulative", "running", "lifetime", "ytd",
    "sum", "mean", "count", "last", "max", "min",
]

_TOKEN_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")


def detect_aggregation_name_signal(column_name: str, keywords: List[str] = DEFAULT_AGGREGATION_KEYWORDS) -> bool:
    """Does this column name look like an aggregated/rolling feature (total_,
    avg_, cumulative_, ..._to_date, etc)?

    Weighting signal, not a gate -- plenty of legitimate, correctly-windowed
    features use these same naming conventions, so this alone never decides
    anything. It's one of three signals combined into an overall severity.

    Matches whole tokens only (split on non-alphanumeric characters), same
    approach as the entity-leakage name signal -- a naive substring check on
    "max" would false-positive on "climax_score", and one on "to_date" would
    miss the far more common underscore-separated "_to_date" suffix while
    still risking false positives like "update_date".
    """
    tokens = [t for t in _TOKEN_SPLIT_RE.split(column_name.lower()) if t]
    keyword_set = set(keywords)

    if any(token in keyword_set for token in tokens):
        return True

    # "_to_date" is a two-token pattern (e.g. "purchases_to_date") rather
    # than a single keyword -- check for "to" immediately followed by "date"
    return any(tokens[i] == "to" and tokens[i + 1] == "date" for i in range(len(tokens) - 1))
