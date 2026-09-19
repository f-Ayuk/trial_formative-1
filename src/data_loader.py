# ============================================================
# src/data_loader.py — Loading and caching (TXT only)
# ============================================================
import pandas as pd
import numpy as np
import gc
from pathlib import Path
from .config import DATA_DIR, CACHE_DIR, MILAN_COLS, OPT_DTYPES
from tqdm.auto import tqdm


def find_files(data_dir=DATA_DIR):
    """Return sorted list of Milan daily .txt files."""
    files = sorted(data_dir.glob("sms-call-internet-mi-*.txt"))
    return [f for f in files if f.is_file()]


def detect_format(filepath):
    """Return (separator, has_header, column_names) for a Milan .txt file.

    Milan files are tab-separated and have no header row.
    We still verify the tab so an unusual copy fails fast.
    """
    with open(filepath) as fh:
        first = fh.readline().rstrip("\n")

    if "\t" not in first:
        raise ValueError(
            f"Expected a tab-separated file, but found no tabs in: {filepath.name}"
        )
    sep = "\t"

    tokens = first.split(sep)
    try:
        float(tokens[0].strip())
        has_header = False
    except (ValueError, IndexError):
        has_header = True

    return sep, has_header, MILAN_COLS


def load_daily(filepath, usecols=None):
    """Load one daily .txt file; prefer Parquet cache.
    Returns a DataFrame with `datetime` always as a column.
    """
    filepath = Path(filepath)
    pq = CACHE_DIR / (filepath.stem + ".parquet")

    if pq.exists():
        df = pd.read_parquet(pq)
        return df[usecols] if usecols else df

    sep, has_header, cols = detect_format(filepath)
    raw_uc = ([("Time Interval" if c == "datetime" else c) for c in usecols]
                if usecols else None)

    df = pd.read_csv(
        filepath,
        sep=sep,
        header=0 if has_header else None,
        names=None if has_header else cols,
        usecols=raw_uc if raw_uc else list(OPT_DTYPES.keys()),
        dtype={k: v for k, v in OPT_DTYPES.items()
                if k in (raw_uc or list(OPT_DTYPES.keys()))},
        engine="python",
    )
    if "Time Interval" in df.columns:
        df["Time Interval"] = pd.to_datetime(df["Time Interval"], unit="ms")
        df = df.rename(columns={"Time Interval": "datetime"})
    return df


def pre_cache_all(files=None):
    """One-time Parquet caching of every daily .txt file."""
    files = files or find_files()
    for f in tqdm(files, desc="Caching to Parquet"):
        pq = CACHE_DIR / (f.stem + ".parquet")
        if pq.exists():
            continue
        df = load_daily(f)
        df.to_parquet(pq, engine="pyarrow", compression="snappy", index=False)
        del df
        gc.collect()


def build_area_series(square_id, files=None, cache_name=None):
    """Extract full-period 10-min series for one area."""
    files = files or find_files()
    cache_name = cache_name or f"ts_square_{square_id}.parquet"
    cache_path = CACHE_DIR / cache_name

    if cache_path.exists():
        return (pd.read_parquet(cache_path)
                    .set_index("datetime")["Internet traffic activity"]
                    .sort_index())

    parts = []
    for f in tqdm(files, desc=f"Square {square_id}"):
        df = load_daily(f, usecols=["datetime", "Square id",
                                    "Internet traffic activity"])
        df = df[df["Square id"] == square_id]
        if len(df):
            parts.append(df)
        del df
        gc.collect()

    full = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    ts = (full.sort_values("datetime")
                .set_index("datetime")["Internet traffic activity"]
                .groupby(level=0).sum().sort_index())
    ts.name = "Internet traffic activity"
    ts.reset_index().to_parquet(cache_path)
    return ts