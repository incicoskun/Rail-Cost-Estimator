"""
process_global_rail.py
Reads raw Excel data and saves a cleaned CSV.

Output: data/processed/global_rail_processed.csv
"""

import pandas as pd
import numpy as np

from src.config import DATA_PROCESSED
from src.load_global_rail import load_raw_projects


# Constants

KEEP_COLS = [
    "Country", "City", "Line", "Phase",
    "Start year", "End year", "RR?",
    "Length", "TunnelPer", "Tunnel",
    "Stations", "Anglo?", "PPP rate",
    "Real cost (2023 dollars)", "Cost/km (2023 dollars)",
]

RENAME_MAP = {
    "Country":                    "country",
    "City":                       "city",
    "Line":                       "line",
    "Phase":                      "phase",
    "Start year":                 "start_year",
    "End year":                   "end_year",
    "RR?":                        "is_regional_rail",
    "Anglo?":                     "is_anglo",
    "PPP rate":                   "ppp_rate",
    "Length":                     "length_km",
    "TunnelPer":                  "tunnel_pct",
    "Tunnel":                     "tunnel_km",
    "Stations":                   "num_stations",
    "Real cost (2023 dollars)":   "real_cost_2023_musd",
    "Cost/km (2023 dollars)":     "cost_per_km_2023_musd",
}

# "slash" → average values like "36/22"
# "year"  → drop non-numeric strings like "not started"
# "float" → direct cast, non-numeric → NaN
NUMERIC_PARSE_STRATEGY = {
    "num_stations":  "slash",
    "start_year":    "year",
    "end_year":      "year",
    "is_rapid_rail": "float",
    "length_km":     "float",
    "tunnel_pct":    "float",
    "tunnel_km":     "float",
    "ppp_rate":      "float",
}


# Parsers

def _parse_slash(val) -> float:
    """
    Converts slash-separated strings to a float average.

    Some projects record station counts as "existing/new" (e.g. "36/22").

    Params: val: any raw value.
    Returns: averaged float or NaN.
    """
    if pd.isna(val):
        return np.nan
    text = str(val).strip()
    if "/" in text:
        parts = text.split("/")
        try:
            return float(np.mean([float(p.strip()) for p in parts]))
        except ValueError:
            print(f"  _parse_slash: '{text}' — cannot convert, set to NaN")
            return np.nan
    try:
        return float(text)
    except ValueError:
        print(f"  _parse_slash: '{text}' — cannot convert, set to NaN")
        return np.nan


def _parse_year(val) -> float:
    """
    Cleans year columns by dropping non-numeric strings.

    Params: val: any raw value.
    Returns: float year or NaN.
    """
    if pd.isna(val):
        return np.nan
    try:
        return float(str(val).strip())
    except ValueError:
        print(f"  _parse_year: '{val}' — not a valid year, set to NaN")
        return np.nan


def _parse_float(val) -> float:
    """
    Standard float cast; returns NaN on failure.

    Params: val: any raw value.
    Returns: float or NaN.
    """
    if pd.isna(val):
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        print(f"  _parse_float: '{val}' — cannot convert, set to NaN")
        return np.nan


_PARSERS = {
    "slash": _parse_slash,
    "year":  _parse_year,
    "float": _parse_float,
}


# Cleaning steps

def _select_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Selects only the required columns from the raw DataFrame."""
    keep = [c for c in KEEP_COLS if c in df.columns]
    return df[keep].copy()


def _rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Renames columns to snake_case using RENAME_MAP."""
    return df.rename(columns=RENAME_MAP)


def _drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows where both country and city are missing."""
    before = len(df)
    df = df.dropna(subset=["country", "city"], how="all")
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows — country and city both missing")
    return df


def _drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows where the target variable is missing."""
    before = len(df)
    df = df.dropna(subset=["cost_per_km_2023_musd"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows — cost_per_km_2023_musd missing")
    return df


def _drop_missing_tunnel(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows where tunnel data is missing."""
    before = len(df)
    df = df.dropna(subset=["tunnel_km", "tunnel_pct"], how="any")
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows — tunnel_km or tunnel_pct missing")
    return df


def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans dirty values in numeric columns and casts to float64.

    Params: df: renamed DataFrame.
    Returns: cleaned DataFrame with float64 numeric columns.
    """
    df = df.copy()
    for col, strategy in NUMERIC_PARSE_STRATEGY.items():
        if col not in df.columns:
            continue
        parser = _PARSERS[strategy]
        nan_before = int(df[col].isna().sum())
        df[col] = df[col].apply(parser).astype("float64")
        new_nans = int(df[col].isna().sum()) - nan_before
        if new_nans > 0:
            print(f"  _clean_numeric: '{col}' — {new_nans} dirty values set to NaN")
    return df


# Main

def process_global_rail() -> pd.DataFrame:
    """
    Runs the full cleaning pipeline and saves the output CSV.

    Steps:
        1. Load raw data
        2. Select columns
        3. Rename columns
        4. Drop incomplete rows
        5. Drop rows with missing target
        6. Drop rows with missing tunnel data
        7. Clean and cast numeric columns

    Returns: cleaned DataFrame.
    """
    df = load_raw_projects()
    print(f"Raw data: {len(df)} rows")

    df = _select_cols(df)
    df = _rename_cols(df)
    df = _drop_incomplete_rows(df)
    df = _drop_missing_target(df)
    df = _drop_missing_tunnel(df)
    df = _clean_numeric(df)

    out_path = DATA_PROCESSED / "global_rail_processed.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} rows, {len(df.columns)} columns)")

    return df


if __name__ == "__main__":
    process_global_rail()