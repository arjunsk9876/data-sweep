from tests.synthetic import make_disjoint_split, make_leaky_split, make_no_entity_structure


def test_leaky_split_has_exact_injected_overlap():
    train_df, test_df = make_leaky_split(overlap_entities=20, n_entities_train=100, n_entities_test_only=100, seed=0)
    train_entities = set(train_df["entity_id"])
    test_entities = set(test_df["entity_id"])
    assert len(train_entities & test_entities) == 20


def test_leaky_split_entities_are_not_row_unique():
    # each entity should appear multiple times, not once per row
    train_df, _ = make_leaky_split(n_train=500, n_entities_train=100, seed=0)
    unique_ratio = train_df["entity_id"].nunique() / len(train_df)
    assert 0.02 < unique_ratio < 0.95


def test_disjoint_split_has_zero_overlap():
    train_df, test_df = make_disjoint_split(seed=0)
    train_entities = set(train_df["entity_id"])
    test_entities = set(test_df["entity_id"])
    assert len(train_entities & test_entities) == 0


def test_disjoint_split_entities_are_not_row_unique():
    train_df, _ = make_disjoint_split(n_train=500, n_entities_train=100, seed=0)
    unique_ratio = train_df["entity_id"].nunique() / len(train_df)
    assert 0.02 < unique_ratio < 0.95


def test_no_entity_structure_has_no_grouping_band_column():
    train_df, test_df = make_no_entity_structure(seed=0)
    for col in train_df.columns:
        ratio = train_df[col].nunique() / len(train_df)
        assert ratio <= 0.02 or ratio >= 0.95, f"{col} unexpectedly falls in the grouping band ({ratio:.2f})"


def test_no_entity_structure_row_ids_are_disjoint_across_splits():
    train_df, test_df = make_no_entity_structure(seed=0)
    assert set(train_df["row_id"]) & set(test_df["row_id"]) == set()


def test_generators_are_deterministic_given_same_seed():
    train_a, test_a = make_leaky_split(seed=42)
    train_b, test_b = make_leaky_split(seed=42)
    assert train_a["entity_id"].tolist() == train_b["entity_id"].tolist()
    assert test_a["entity_id"].tolist() == test_b["entity_id"].tolist()
