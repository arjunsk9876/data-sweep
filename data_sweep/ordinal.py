from typing import Optional

import pandas as pd

ORDINAL_SCALES: list[list[str]] = [
    ["never", "rarely", "sometimes", "often", "always"],
    ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
    ["disagree", "neutral", "agree"],
    ["low", "medium", "high"],
    ["small", "medium", "large"],
    ["poor", "fair", "good", "excellent"],
    ["bad", "average", "good"],
]


def match_ordinal_scale(values: pd.Series) -> Optional[list[str]]:
    lowered = {str(v).strip().lower() for v in values}
    if len(lowered) < 2:
        return None
    for scale in ORDINAL_SCALES:
        if lowered <= set(scale):
            return scale
    return None


def ordinal_mapping(values: pd.Series, scale: list[str]) -> dict:
    return {v: scale.index(str(v).strip().lower()) for v in values.unique()}
