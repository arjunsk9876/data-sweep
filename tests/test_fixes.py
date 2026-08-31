import sys

import pytest

from data_sweep.entity_leakage.findings import EntityLeakageFinding
from data_sweep.entity_leakage.fixes import (
    NoFindingsError,
    SklearnMissingError,
    check_sklearn_available,
    generate_fix_code,
)


def _finding(candidate_key="entity_id", uniqueness_ratio=0.2, overlap_pct=20.0, severity="high", mode="two_file", example_values=None):
    return EntityLeakageFinding(
        candidate_key=candidate_key,
        uniqueness_ratio=uniqueness_ratio,
        overlap_pct=overlap_pct,
        severity=severity,
        mode=mode,
        example_values=example_values if example_values is not None else [],
    )


def test_two_file_mode_uses_group_shuffle_split():
    code = generate_fix_code([_finding(mode="two_file")])
    assert "GroupShuffleSplit" in code
    assert "GroupKFold" not in code


def test_single_file_mode_uses_group_k_fold():
    code = generate_fix_code([_finding(mode="single_file")])
    assert "GroupKFold" in code
    assert "GroupShuffleSplit" not in code


def test_substitutes_real_candidate_key():
    code = generate_fix_code([_finding(candidate_key="customer_id")])
    assert "customer_id" in code
    assert "{candidate_key}" not in code


def test_column_name_with_quote_produces_valid_python():
    code = generate_fix_code([_finding(candidate_key="cust's_id")])
    compile(code, "<generated-fix>", "exec")  # raises SyntaxError if malformed


def test_no_findings_raises():
    with pytest.raises(NoFindingsError):
        generate_fix_code([])


def test_no_alt_note_for_single_finding():
    code = generate_fix_code([_finding()])
    assert "# Note:" not in code


def test_picks_highest_severity_as_primary():
    low = _finding(candidate_key="low_col", severity="low", overlap_pct=3.0)
    high = _finding(candidate_key="high_col", severity="high", overlap_pct=25.0)
    code = generate_fix_code([low, high])
    assert "groups=df['high_col']" in code
    assert "groups=df['low_col']" not in code


def test_alternate_candidates_get_two_file_overlap_note():
    primary = _finding(candidate_key="primary_col", severity="high", overlap_pct=25.0)
    alt = _finding(candidate_key="alt_col", severity="medium", overlap_pct=4.2, mode="two_file")
    code = generate_fix_code([primary, alt])
    assert "# Note: 'alt_col' also showed 4.2% overlap" in code
    assert "groups=df['primary_col']" in code


def test_alternate_candidates_get_single_file_uniqueness_note():
    primary = _finding(candidate_key="primary_col", severity="high", mode="single_file")
    alt = _finding(candidate_key="alt_col", severity="low", mode="single_file", uniqueness_ratio=0.111)
    code = generate_fix_code([primary, alt])
    assert "# Note: 'alt_col' is also a candidate entity/group key (uniqueness ratio 0.111)" in code


def test_multiple_alternates_all_noted():
    primary = _finding(candidate_key="primary_col", severity="high")
    alt1 = _finding(candidate_key="alt1", severity="medium")
    alt2 = _finding(candidate_key="alt2", severity="low")
    code = generate_fix_code([primary, alt1, alt2])
    assert "alt1" in code
    assert "alt2" in code
    assert code.count("# Note:") == 2


def test_generated_code_is_syntactically_valid():
    code = generate_fix_code([_finding()])
    compile(code, "<generated-fix>", "exec")


def test_check_sklearn_available_does_not_raise_when_installed():
    check_sklearn_available()  # scikit-learn is a dev dependency -- should be importable here


def test_check_sklearn_available_raises_clean_error_when_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn", None)
    with pytest.raises(SklearnMissingError, match="pip install scikit-learn"):
        check_sklearn_available()


def test_generate_fix_code_raises_when_sklearn_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn", None)
    with pytest.raises(SklearnMissingError):
        generate_fix_code([_finding()])


def test_generate_fix_code_checks_sklearn_after_empty_findings_check():
    # empty findings should raise NoFindingsError regardless of sklearn
    # availability -- there's nothing to generate a fix for either way
    with pytest.raises(NoFindingsError):
        generate_fix_code([])
