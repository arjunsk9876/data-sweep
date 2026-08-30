import re
from typing import List

DEFAULT_NAME_KEYWORDS = ["id", "key", "uuid", "guid", "no", "num", "number", "code"]

_TOKEN_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")


def detect_name_signal(column_name: str, keywords: List[str] = DEFAULT_NAME_KEYWORDS) -> bool:
    """Does this column name hint at being an id/key (id, key, uuid, _no, number, code)?

    Weighting boost, not a gate — must also work fine (return False, no
    penalty) on anonymized/renamed columns with no naming hints at all; the
    uniqueness gate in keys.py is what actually decides candidacy.

    Matches whole tokens only (split on non-alphanumeric characters), not
    bare substrings — a naive substring check on "id" would false-positive
    on "paid", "void", "valid", and plenty of other ordinary words.
    """
    tokens = [t for t in _TOKEN_SPLIT_RE.split(column_name.lower()) if t]
    keyword_set = set(keywords)
    return any(token in keyword_set for token in tokens)
