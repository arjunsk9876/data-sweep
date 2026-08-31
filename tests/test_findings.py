from data_sweep.entity_leakage.findings import EntityLeakageFinding


def _finding(**overrides):
    defaults = dict(
        candidate_key="entity_id",
        uniqueness_ratio=0.2,
        overlap_pct=20.0,
        severity="high",
        mode="two_file",
        example_values=["E1", "E2"],
    )
    defaults.update(overrides)
    return EntityLeakageFinding(**defaults)


def test_construction_holds_all_fields():
    f = _finding()
    assert f.candidate_key == "entity_id"
    assert f.uniqueness_ratio == 0.2
    assert f.overlap_pct == 20.0
    assert f.severity == "high"
    assert f.mode == "two_file"
    assert f.example_values == ["E1", "E2"]


def test_equality_by_value():
    assert _finding() == _finding()
    assert _finding(candidate_key="other") != _finding()


def test_single_file_mode_allows_zero_overlap_pct():
    f = _finding(mode="single_file", overlap_pct=0.0, example_values=[])
    assert f.mode == "single_file"
    assert f.overlap_pct == 0.0
    assert f.example_values == []
