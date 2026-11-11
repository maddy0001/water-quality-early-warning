"""Turn any raw table into the common stream format (see ``ewsv2.core.types``)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ewsv2.core.types import CLASS_COL, EVENT_COL, TIME_COL

log = logging.getLogger(__name__)

_TIME_NAMES = re.compile(r"^(time|timestamp|datetime|date_time|date|zeit)$", re.IGNORECASE)
_INDEX_NAMES = re.compile(r"^(unnamed: ?\d+|index|x|\.\.\.1|id|row)$", re.IGNORECASE)


class DataFormatError(ValueError):
    """The file does not look like a sensor stream."""


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV (any common separator) or R ``.rda/.rds`` into a DataFrame."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".rda", ".rdata", ".rds"):
        try:
            import pyreadr
        except ImportError as exc:  # pragma: no cover - depends on install
            raise DataFormatError(
                f"{path.name} is an R data file; install pyreadr (pip install pyreadr)"
            ) from exc
        result = pyreadr.read_r(str(path))
        if not result:
            raise DataFormatError(f"{path} contains no tables")
        return next(iter(result.values()))
    # sep=None lets pandas sniff "," ";" or tab; GECCO files have used more than one.
    return pd.read_csv(path, sep=None, engine="python")


def _find_time_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if _TIME_NAMES.match(str(col).strip()):
            return col
    for col in df.columns:  # fall back: first column that parses as dates
        if df[col].dtype == object or np.issubdtype(df[col].dtype, np.datetime64):
            parsed = pd.to_datetime(df[col].head(50), errors="coerce")
            if parsed.notna().mean() > 0.9:
                return col
    raise DataFormatError(f"No time column found among {list(df.columns)}")


def _find_event_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if str(col).strip().lower() in ("event", "events", "label", "anomaly"):
            return col
    return None


def _to_binary(values: pd.Series) -> pd.Series:
    """TRUE/FALSE, true/false, 1/0, 1.0/0.0 -> 0/1 ints (missing -> 0)."""
    if values.dtype == bool:
        return values.astype(int)
    mapped = values.astype(str).str.strip().str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0, "t": 1, "f": 0}
    )
    unknown = mapped.isna() & values.notna()
    if unknown.any():
        examples = values[unknown].unique()[:5]
        raise DataFormatError(f"EVENT column has values that are not 0/1 or TRUE/FALSE: {examples}")
    return mapped.fillna(0).astype(int)


def _as_number(col: pd.Series) -> pd.Series:
    """Text that is really numbers ("8.1" or German "8,1") -> floats; otherwise unchanged."""
    present = col.dropna()
    if present.empty:
        return col
    text = present.astype(str).str.strip()
    for candidate in (text, text.str.replace(",", ".", regex=False)):
        parsed = pd.to_numeric(candidate, errors="coerce")
        if parsed.notna().mean() >= 0.9:
            out = pd.Series(np.nan, index=col.index, dtype=float)
            out[present.index] = parsed.to_numpy()
            return out
    return col


def standardise(raw: pd.DataFrame, sensors: list[str] | None = None) -> tuple[pd.DataFrame, list[str], dict]:
    """Return (frame, sensor_names, notes) in the common format.

    * time column -> sorted, unique ``DatetimeIndex`` named ``time``
    * EVENT -> 0/1 int; LABEL_CLASS copied from EVENT (1 = water event)
    * sensors -> float columns; if ``sensors`` is None, every numeric column
      that is not time, label or a row index is treated as a sensor
    """
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    notes: dict = {}

    time_col = _find_time_column(df)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    bad_time = int(df[time_col].isna().sum())
    if bad_time:
        notes["rows_with_unparseable_time_dropped"] = bad_time
        df = df[df[time_col].notna()]

    event_col = _find_event_column(df)
    if event_col is not None:
        df[EVENT_COL] = _to_binary(df[event_col])
        if event_col != EVENT_COL:
            df = df.drop(columns=[event_col])
    else:
        df[EVENT_COL] = 0
        notes["no_event_column"] = True

    skip = {time_col, EVENT_COL, CLASS_COL}
    for c in df.columns:
        if c not in skip and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = _as_number(df[c])

    if sensors is None:
        sensors = [
            c for c in df.columns
            if c not in skip and not _INDEX_NAMES.match(c) and pd.api.types.is_numeric_dtype(df[c])
        ]
    missing = [s for s in sensors if s not in df.columns]
    if missing:
        raise DataFormatError(f"Sensor column(s) not found: {missing}. Columns: {list(df.columns)}")
    if not sensors:
        raise DataFormatError("No numeric sensor columns found")

    df = df[[time_col, *sensors, EVENT_COL]].rename(columns={time_col: TIME_COL})
    for s in sensors:
        df[s] = pd.to_numeric(df[s], errors="coerce").astype(float)

    df = df.sort_values(TIME_COL, kind="stable")
    dup = int(df[TIME_COL].duplicated().sum())
    if dup:
        notes["duplicate_timestamps_dropped"] = dup
        df = df.drop_duplicates(subset=TIME_COL, keep="first")

    df = df.set_index(TIME_COL)
    df[CLASS_COL] = df[EVENT_COL].astype(int)  # 1 = water event; faults added in S4
    return df, list(sensors), notes
