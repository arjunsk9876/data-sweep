import pytest

from data_sweep.entity_leakage.io import DatasetLoadError, load_datasets


def test_loads_train_only(tmp_path):
    train_path = tmp_path / "train.csv"
    train_path.write_text("a,b\n1,x\n2,y\n")

    train_df, test_df = load_datasets(str(train_path))

    assert list(train_df.columns) == ["a", "b"]
    assert len(train_df) == 2
    assert test_df is None


def test_loads_train_and_test(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_path.write_text("a,b\n1,x\n2,y\n")
    test_path.write_text("a,b\n3,z\n")

    train_df, test_df = load_datasets(str(train_path), str(test_path))

    assert len(train_df) == 2
    assert test_df is not None
    assert len(test_df) == 1


def test_missing_train_file_raises_clean_error(tmp_path):
    with pytest.raises(DatasetLoadError, match="file not found"):
        load_datasets(str(tmp_path / "does_not_exist.csv"))


def test_missing_test_file_raises_clean_error(tmp_path):
    train_path = tmp_path / "train.csv"
    train_path.write_text("a,b\n1,x\n")

    with pytest.raises(DatasetLoadError, match="file not found"):
        load_datasets(str(train_path), str(tmp_path / "does_not_exist.csv"))


def test_empty_file_raises_clean_error(tmp_path):
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("")

    with pytest.raises(DatasetLoadError, match="empty"):
        load_datasets(str(empty_path))


def test_malformed_csv_raises_clean_error(tmp_path):
    bad_path = tmp_path / "bad.csv"
    # inconsistent column counts across rows
    bad_path.write_text("a,b,c\n1,2\n3,4,5,6\n")

    with pytest.raises(DatasetLoadError):
        load_datasets(str(bad_path))


def test_directory_path_raises_clean_error(tmp_path):
    with pytest.raises(DatasetLoadError, match="directory"):
        load_datasets(str(tmp_path))


def test_non_utf8_binary_file_raises_clean_error(tmp_path):
    bad_path = tmp_path / "binary.csv"
    bad_path.write_bytes(bytes(range(256)) * 4)

    with pytest.raises(DatasetLoadError, match="couldn't read"):
        load_datasets(str(bad_path))


def test_permission_denied_raises_clean_error(tmp_path):
    unreadable_path = tmp_path / "unreadable.csv"
    unreadable_path.write_text("a,b\n1,x\n")
    unreadable_path.chmod(0o000)
    try:
        with pytest.raises(DatasetLoadError, match="couldn't read"):
            load_datasets(str(unreadable_path))
    finally:
        # restore permissions so pytest can clean up tmp_path afterward
        unreadable_path.chmod(0o644)
