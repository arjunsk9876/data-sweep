"""Synthetic train/test dataset generators with known, ground-truth entity structure.

Used by the entity-leakage detector tests to check both true positives (a
real injected leak gets found) and true negatives (no leak, or no entity
structure at all, produces no findings).
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
