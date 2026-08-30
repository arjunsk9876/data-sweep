from data_sweep.entity_leakage.name_signal import detect_name_signal


def test_id_suffix_detected():
    assert detect_name_signal("household_id") is True


def test_key_suffix_detected():
    assert detect_name_signal("customer_key") is True


def test_uuid_detected():
    assert detect_name_signal("session_uuid") is True


def test_no_suffix_detected():
    assert detect_name_signal("account_no") is True


def test_number_detected():
    assert detect_name_signal("invoice_number") is True


def test_code_detected():
    assert detect_name_signal("device_code") is True


def test_case_insensitive():
    assert detect_name_signal("HOUSEHOLD_ID") is True


def test_bare_keyword_detected():
    assert detect_name_signal("id") is True


def test_unrelated_name_not_detected():
    assert detect_name_signal("temperature") is False


def test_substring_false_positives_avoided():
    # these all *contain* "id" as a substring but aren't id-shaped tokens —
    # a naive substring match would wrongly fire on every one of these
    assert detect_name_signal("paid") is False
    assert detect_name_signal("valid") is False
    assert detect_name_signal("width") is False
    assert detect_name_signal("void") is False


def test_anonymized_column_names_not_detected():
    # PRD: must work fine (no penalty, just no boost) on renamed/anonymized
    # columns with no naming hints at all
    assert detect_name_signal("col_7") is False
    assert detect_name_signal("x1") is False
    assert detect_name_signal("feature_42") is False
