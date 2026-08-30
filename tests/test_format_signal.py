import pandas as pd

from data_sweep.entity_leakage.format_signal import detect_format_signal


def test_uuid_values_detected():
    values = pd.Series([
        "550e8400-e29b-41d4-a716-446655440000",
        "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
        "6ba7b811-9dad-11d1-80b4-00c04fd430c8",
    ])
    assert detect_format_signal(values) is True


def test_zero_padded_numbers_detected():
    values = pd.Series(["00042", "00123", "00007", "00891"])
    assert detect_format_signal(values) is True


def test_hash_like_hex_detected():
    values = pd.Series([
        "9e107d9d372bb6826bd81d3542a419d6",
        "e4d909c290d0fb1ca068ffaddf22cbd0",
        "d41d8cd98f00b204e9800998ecf8427e",
    ])
    assert detect_format_signal(values) is True


def test_alphanumeric_codes_detected():
    values = pd.Series(["AB1234", "X9F2D1", "ZZ0099", "QR5566"])
    assert detect_format_signal(values) is True


def test_plain_integers_not_detected():
    # PRD: format signal is optional, a plain integer id should NOT need it
    values = pd.Series(["1", "2", "3", "42", "1007"])
    assert detect_format_signal(values) is False


def test_plain_words_not_detected():
    values = pd.Series(["red", "green", "blue", "yellow"])
    assert detect_format_signal(values) is False


def test_empty_series_not_detected():
    assert detect_format_signal(pd.Series([], dtype=object)) is False


def test_below_threshold_match_ratio_not_detected():
    # only 2/10 look ID-like -> below the 0.9 default threshold
    values = pd.Series(["AB1234"] * 2 + ["plain"] * 8)
    assert detect_format_signal(values) is False


def test_above_threshold_match_ratio_detected():
    # 9/10 look ID-like -> above the 0.9 default threshold
    values = pd.Series(["AB1234"] * 9 + ["plain"])
    assert detect_format_signal(values) is True
