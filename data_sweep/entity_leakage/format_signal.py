import re

import pandas as pd

DEFAULT_MIN_MATCH_RATIO = 0.9

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
_HASH_RE = re.compile(r"^[0-9a-f]{16,64}$", re.IGNORECASE)
_ZERO_PADDED_RE = re.compile(r"^0\d+$")
_ALPHANUMERIC_CODE_RE = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9_-]{4,}$")


def _looks_id_like(value: str) -> bool:
    return bool(
        _UUID_RE.match(value)
        or _HASH_RE.match(value)
        or _ZERO_PADDED_RE.match(value)
        or _ALPHANUMERIC_CODE_RE.match(value)
    )


def detect_format_signal(series: pd.Series, min_match_ratio: float = DEFAULT_MIN_MATCH_RATIO) -> bool:
    """Does this column look ID-formatted (UUID, hash, zero-padded, alphanumeric code)?

    Secondary signal, not a requirement — a plain integer entity id won't
    match any of these patterns and that's fine; this only ever adds a
    ranking boost on top of the uniqueness gate, never gates on its own.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    match_ratio = non_null.astype(str).map(_looks_id_like).mean()
    return bool(match_ratio >= min_match_ratio)
