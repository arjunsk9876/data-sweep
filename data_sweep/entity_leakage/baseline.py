from dataclasses import dataclass
from typing import Optional

import pandas as pd

MIN_ROWS_FOR_PREDICTIVENESS = 10


@dataclass
class PredictivenessResult:
    score: float
    metric: str  # "auc" (binary target) or "r2" (continuous target)


def compute_predictiveness(feature: pd.Series, target: pd.Series) -> Optional[PredictivenessResult]:
    """How predictive is this single feature of the target, on its own?

    No model fitting, no train/test split -- this is a cheap, dependency-free
    single-feature baseline, not a real model. Reused as a leakage-signal
    building block: temporal leakage's Signal 3 (a feature unusually
    predictive on its own is suspicious), and later basic feature-target
    leakage too, per the roadmap.

    Returns None when a score can't be meaningfully computed (too few
    complete rows, a constant feature with no discriminative power, or a
    target shape this doesn't yet support) rather than fabricating a
    misleading number.
    """
    aligned = pd.DataFrame({"feature": feature, "target": target}).dropna()
    if len(aligned) < MIN_ROWS_FOR_PREDICTIVENESS:
        return None

    if aligned["target"].nunique() == 2:
        return _binary_auc(aligned)

    return None


def _binary_auc(aligned: pd.DataFrame) -> Optional[PredictivenessResult]:
    if aligned["feature"].nunique() < 2:
        return None  # constant feature -- nothing to rank, no discriminative power

    labels = sorted(aligned["target"].unique())
    positive_label = labels[-1]  # the larger of the two labels is "positive" (1 in a 0/1 encoding)

    n_pos = int((aligned["target"] == positive_label).sum())
    n_neg = len(aligned) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    ranks = aligned["feature"].rank(method="average")
    rank_sum_pos = ranks[aligned["target"] == positive_label].sum()
    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2
    auc = u_stat / (n_pos * n_neg)

    # direction-agnostic: a feature that's inversely predictive (AUC near 0)
    # is exactly as suspicious as one that's directly predictive (AUC near
    # 1) for leakage-flagging purposes, so report discriminative power, not
    # which way it points
    score = max(auc, 1 - auc)
    return PredictivenessResult(score=float(score), metric="auc")
