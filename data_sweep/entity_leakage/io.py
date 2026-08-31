from typing import Optional, Tuple

import pandas as pd


class DatasetLoadError(Exception):
    """Raised when a dataset file can't be loaded as a usable CSV."""


def load_datasets(train_path: str, test_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load the training file, and the test file if one was given.

    Raises DatasetLoadError with a plain-language message on any failure —
    callers (the CLI) turn that into a clean error message and exit, rather
    than a raw traceback.
    """
    train_df = _load_csv(train_path)
    test_df = _load_csv(test_path) if test_path else None
    return train_df, test_df


def _load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise DatasetLoadError(f"file not found: {path}") from None
    except IsADirectoryError:
        raise DatasetLoadError(f"expected a file but found a directory: {path}") from None
    except pd.errors.EmptyDataError:
        raise DatasetLoadError(f"file is empty: {path}") from None
    except pd.errors.ParserError as e:
        raise DatasetLoadError(f"couldn't parse {path} as CSV: {e}") from None
    except UnicodeDecodeError:
        raise DatasetLoadError(f"couldn't read {path} as text -- is this actually a CSV file?") from None
    except OSError as e:
        # catch-all for anything else the filesystem can throw (permission
        # denied, a broken symlink, etc.) so a bad path never surfaces a raw
        # traceback, even one this code didn't anticipate specifically
        raise DatasetLoadError(f"couldn't read {path}: {e}") from None
