"""Synthetic dataset generators with known, guaranteed ground truth.

Used by the leakage detector tests (entity leakage's train/test pairs, and
temporal leakage's single-file feature datasets) to check both true
positives (a real injected leak gets found) and true negatives (no leak, or
no leak-shaped structure at all, produces no findings).
"""
from typing import List, Optional

import numpy as np
import pandas as pd


def make_leaky_split(
    n_train: int = 500,
    n_test: int = 200,
    n_entities_train: int = 100,
    n_entities_test_only: int = 100,
    overlap_entities: int = 20,
    entity_col: str = "entity_id",
    seed: int = 0,
) -> tuple:
    """Train/test pair with a real, injected entity leak.

    `overlap_entities` distinct `entity_col` values are guaranteed to appear
    in both splits — that's the ground-truth leak. Each entity has multiple
    rows (grouping-band cardinality: never close to 100% unique, never close
    to constant).
    """
    rng = np.random.RandomState(seed)

    train_pool = [f"E{i}" for i in range(n_entities_train)]
    shared = train_pool[:overlap_entities]
    test_only_pool = [f"T{i}" for i in range(n_entities_test_only)]
    test_pool = shared + test_only_pool

    train_df = _make_frame(rng, n_train, entity_col, train_pool, must_include=shared)
    test_df = _make_frame(rng, n_test, entity_col, test_pool, must_include=shared)
    return train_df, test_df


def make_disjoint_split(
    n_train: int = 500,
    n_test: int = 200,
    n_entities_train: int = 100,
    n_entities_test: int = 100,
    entity_col: str = "entity_id",
    seed: int = 0,
) -> tuple:
    """Train/test pair with a real entity column, but zero overlap.

    A genuine grouping-band candidate key exists in both files, but the two
    entity pools are fully disjoint — ground truth: no leak. Tests that the
    overlap check doesn't false-positive just because a candidate key exists.
    """
    rng = np.random.RandomState(seed)

    train_pool = [f"E{i}" for i in range(n_entities_train)]
    test_pool = [f"T{i}" for i in range(n_entities_test)]

    train_df = _make_frame(rng, n_train, entity_col, train_pool)
    test_df = _make_frame(rng, n_test, entity_col, test_pool)
    return train_df, test_df


def make_no_entity_structure(
    n_train: int = 500,
    n_test: int = 200,
    seed: int = 0,
) -> tuple:
    """Train/test pair with no grouping-band column at all.

    Every column is either ~100% unique (a true row id) or low-cardinality
    (a plain categorical) — nothing should score as a candidate entity key.
    Ground truth: no candidate keys, no leakage findings.
    """
    rng = np.random.RandomState(seed)

    train_df = pd.DataFrame({
        "row_id": [f"R{i}" for i in range(n_train)],
        "category": rng.choice(["a", "b", "c"], n_train),
        "value": rng.normal(0, 1, n_train),
    })
    test_df = pd.DataFrame({
        "row_id": [f"R{i}" for i in range(n_train, n_train + n_test)],
        "category": rng.choice(["a", "b", "c"], n_test),
        "value": rng.normal(0, 1, n_test),
    })
    return train_df, test_df


def make_temporal_leak_dataset(
    n_rows: int = 1000,
    seed: int = 0,
    elapsed_leak_strength: float = 0.6,
    target_leak_boost: float = 90.0,
) -> pd.DataFrame:
    """Single-file dataset with a known-good and a known-leaked aggregate.

    Both `total_purchases` (leaked) and `total_purchases_windowed` (clean
    control) are built from the same underlying `true_rate` signal, so they
    start out comparably informative -- the leaked one is then additionally
    inflated by `elapsed_days` (the gap between `snapshot_date` and
    `cancel_date`) and by `target` itself, simulating a feature that
    accidentally kept accumulating past the label event instead of being
    cut off at `snapshot_date`.

    Pass elapsed_leak_strength=0, target_leak_boost=0 to get an all-clean
    dataset instead (both features legitimately windowed) -- useful for
    false-positive testing without a second generator function.
    """
    rng = np.random.RandomState(seed)

    true_rate = np.clip(rng.normal(5, 1.5, n_rows), 0.5, None)

    # genuine, modest relationship: a higher true purchase rate makes churn
    # somewhat less likely -- so even the clean control feature carries some
    # real signal, it just shouldn't be *unusually* predictive on its own
    prob_churn = 1 / (1 + np.exp(0.15 * (true_rate - 5)))
    target = (rng.uniform(0, 1, n_rows) < prob_churn).astype(int)

    elapsed_days = rng.uniform(1, 200, n_rows)
    snapshot_date = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.uniform(0, 365, n_rows), unit="D")
    cancel_date = snapshot_date + pd.to_timedelta(elapsed_days, unit="D")

    fixed_window_days = 30
    clean_feature = true_rate * fixed_window_days + rng.normal(0, 3, n_rows)

    leaked_feature = (
        true_rate * fixed_window_days
        + elapsed_leak_strength * elapsed_days
        + target_leak_boost * target
        + rng.normal(0, 2, n_rows)
    )

    return pd.DataFrame({
        "snapshot_date": snapshot_date,
        "cancel_date": cancel_date,
        "target": target,
        "total_purchases": leaked_feature,
        "total_purchases_windowed": clean_feature,
    })


def _make_frame(
    rng: np.random.RandomState,
    n_rows: int,
    entity_col: str,
    entity_pool: List[str],
    must_include: Optional[List[str]] = None,
) -> pd.DataFrame:
    entities = rng.choice(entity_pool, size=n_rows).tolist() if n_rows > 0 else []

    if must_include:
        # random sampling doesn't guarantee every pool value gets drawn in a
        # finite sample, so force the required values into fixed slots
        assert len(must_include) <= n_rows, "must_include can't exceed n_rows"
        for i, value in enumerate(must_include):
            entities[i] = value
        rng.shuffle(entities)

    return pd.DataFrame({
        entity_col: entities,
        "feature_a": rng.normal(0, 1, n_rows),
        "feature_b": rng.choice(["x", "y", "z"], n_rows),
    })
