from data_sweep.entity_leakage.findings import EntityLeakageFinding, from_candidate_keys
from data_sweep.entity_leakage.keys import CandidateKey


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


def _candidate(column="entity_id", uniqueness_ratio=0.2, score=1.0, signals=None):
    return CandidateKey(
        column=column,
        uniqueness_ratio=uniqueness_ratio,
        score=score,
        signals=signals if signals is not None else ["uniqueness"],
    )


def test_from_candidate_keys_maps_fields():
    candidates = [_candidate(column="entity_id", uniqueness_ratio=0.183, score=1.25, signals=["uniqueness", "format", "name"])]
    findings = from_candidate_keys(candidates)

    assert len(findings) == 1
    f = findings[0]
    assert f.candidate_key == "entity_id"
    assert f.uniqueness_ratio == 0.183
    assert f.overlap_pct == 0.0
    assert f.mode == "single_file"
    assert f.example_values == []


def test_from_candidate_keys_severity_boundaries():
    high = _candidate(column="a", score=1.25)          # both signals
    just_below_high = _candidate(column="b", score=1.15)  # one signal
    medium = _candidate(column="c", score=1.10)          # one signal
    just_below_medium = _candidate(column="d", score=1.0)  # uniqueness only

    mapped = {f.candidate_key: f.severity for f in from_candidate_keys([high, just_below_high, medium, just_below_medium])}
    assert mapped["a"] == "high"
    assert mapped["b"] == "medium"
    assert mapped["c"] == "medium"
    assert mapped["d"] == "low"


def test_from_candidate_keys_empty_list():
    assert from_candidate_keys([]) == []
