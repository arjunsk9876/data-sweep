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

    A binary target (exactly 2 distinct values) is scored as AUC; any other
    numeric target is treated as continuous and scored as R^2. A non-numeric
    target with more than 2 categories (true multi-class) isn't supported
    yet and returns None -- proper one-vs-rest scoring is more machinery
    than this phase needs.

    Returns None whenever a score can't be meaningfully computed (too few
    complete rows, a constant feature with no discriminative power, or an
    unsupported target shape) rather than fabricating a misleading number.
    """
    aligned = pd.DataFrame({"feature": feature, "target": target}).dropna()
    if len(aligned) < MIN_ROWS_FOR_PREDICTIVENESS:
        return None

    if aligned["target"].nunique() == 2:
        return _binary_auc(aligned)

    if pd.api.types.is_numeric_dtype(aligned["target"]):
        return _continuous_r2(aligned)

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


def _continuous_r2(aligned: pd.DataFrame) -> Optional[PredictivenessResult]:
    if not pd.api.types.is_numeric_dtype(aligned["feature"]):
        return None  # can't correlate a non-numeric feature with a continuous target
    if aligned["feature"].nunique() < 2 or aligned["target"].nunique() < 2:
        return None  # a constant series has no variance to correlate

    corr = aligned["feature"].corr(aligned["target"])
    if pd.isna(corr):
        return None

    return PredictivenessResult(score=float(corr ** 2), metric="r2")
