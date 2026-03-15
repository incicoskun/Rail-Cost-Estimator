"""
process_fta.py
Full processing pipeline for FTA cost breakdown data.

Output: data/processed/fta_processed.csv
"""

import pandas as pd

from src.config import DATA_PROCESSED


# Constants

DOLLAR_COLS = [
    "Project Cost", "Cost per Mile",
    "Hard Costs + ROW", "Hard Costs", "Soft Costs",
    "Avg Cost per Vehicle", "Avg Cost per Station",
    "Vehicle Costs", "ROW Costs", "Systems Costs",
    "Sitework Costs", "Facilities Costs", "Station Costs", "Guideway Costs",
    "Soft Costs per Mile", "ROW Costs per Mile",
    "System Costs per Mile", "Sitework Costs per Mile",
    "Facility Costs per Mile", "Guideway Costs per Mile",
]

# Columns that arrive as strings with "%" suffix → need strip + /100
PCT_COLS_STR: list[str] = []  

# Columns that arrive as float64 decimals (0–1) → validate only
PCT_COLS_FLOAT = [
    "Tunnel %", "Soft/ Hard Percent",
    "Soft Costs %", "Vehicle Costs %", "ROW Costs %",
    "Systems Costs %", "Sitework Costs %", "Facilities Costs %",
    "Station Costs %", "Guideway Costs %",
]

RENAME_MAP = {
    "Project\xa0(All costs have been adjusted"
    " for inflation in 2021 dollars)": "project",
    "Mode":                  "mode",
    "Mode.1":                "mode_full",
    "locCity":               "city",
    "Year":                  "year",
    "Length":                "length",
    "at grade":              "atgrade_miles",
    "above grade":           "elevated_miles",
    "below grade":           "below_miles",
    "Tunnel Miles":          "tunnel_miles",
    "Tunnel %":              "tunnel_pct",
    "Project Cost":          "project_cost",
    "Cost per Mile":         "cost_per_mile",
    "Hard Costs + ROW":      "hard_costs_row",
    "Hard Costs":            "hard_costs",
    "Soft Costs":            "soft_costs",
    "Soft/ Hard Percent":    "soft_hard_pct",
    "Vehicles":              "vehicles",
    "Avg Cost per Vehicle":  "avg_cost_per_vehicle",
    "Stations":              "stations",
    "Avg Cost per Station":  "avg_cost_per_station",
    "Vehicle Costs":         "vehicle_costs",
    "ROW Costs":             "row_costs",
    "Systems Costs":         "systems_costs",
    "Sitework Costs":        "sitework_costs",
    "Facilities Costs":      "facilities_costs",
    "Station Costs":         "station_costs",
    "Guideway Costs":        "guideway_costs",
    "Soft Costs %":          "soft_costs_pct",
    "Vehicle Costs %":       "vehicle_costs_pct",
    "ROW Costs %":           "row_costs_pct",
    "Systems Costs %":       "systems_costs_pct",
    "Sitework Costs %":      "sitework_costs_pct",
    "Facilities Costs %":    "facilities_costs_pct",
    "Station Costs %":       "station_costs_pct",
    "Guideway Costs %":      "guideway_costs_pct",
    "Soft Costs per Mile":   "soft_costs_per_mile",
    "ROW Costs per Mile":    "row_costs_per_mile",
    "System Costs per Mile": "system_costs_per_mile",
    "Sitework Costs per Mile": "sitework_costs_per_mile",
    "Facility Costs per Mile": "facility_costs_per_mile",
    "Guideway Costs per Mile": "guideway_costs_per_mile",
}


# Cleaning helpers

def _clean_dollar_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Removes $ and commas from dollar columns, converts to float."""
    for col in DOLLAR_COLS:
        if col not in df.columns:
            continue
        cleaned = (
            df[col].astype(str)
            .str.replace(r"[$,]", "", regex=True)
            .str.strip()
        )
        df[col] = pd.to_numeric(cleaned, errors="coerce")
        n_null = df[col].isna().sum()
        if n_null > 0:
            print(f"  '{col}': {n_null} nulls after coercion")
    return df


def _clean_pct_cols_str(df: pd.DataFrame) -> pd.DataFrame:
    """Strips % suffix and converts to decimal (0–1)."""
    for col in PCT_COLS_STR:
        if col not in df.columns:
            continue
        cleaned = (
            df[col].astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(cleaned, errors="coerce")
        if df[col].dropna().max() > 1:
            df[col] = df[col] / 100
        n_null = df[col].isna().sum()
        if n_null > 0:
            print(f"  '{col}': {n_null} nulls after coercion")
    return df


def _validate_pct_cols_float(df: pd.DataFrame) -> pd.DataFrame:
    """Validates that float pct columns are within 0–1 range."""
    for col in PCT_COLS_FLOAT:
        if col not in df.columns:
            continue
        out_of_range = df[col].dropna()
        out_of_range = out_of_range[(out_of_range < 0) | (out_of_range > 1)]
        if not out_of_range.empty:
            print(f"  '{col}': {len(out_of_range)} values outside 0–1 range")
    return df


def _inflate_to_2023(df: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    """Multiplies dollar columns by the 2021→2023 CPI multiplier."""
    for col in DOLLAR_COLS:
        if col in df.columns:
            df[col + "_2023"] = df[col] * multiplier
    return df


def _rename_to_snake(df: pd.DataFrame) -> pd.DataFrame:
    """Renames all columns to snake_case using RENAME_MAP."""
    return df.rename(columns=RENAME_MAP)


# Public pipeline

def _get_inflation_multiplier(cpi: pd.DataFrame) -> float:
    """
    Returns the multiplier to convert 2021 dollars to 2023 dollars.

    Params: cpi: DataFrame with Year as index and Index_2023 column.
    Returns: inflation multiplier as float.
    """
    return float(cpi.loc[2021, "Index_2023"])


def process_fta(df: pd.DataFrame, cpi: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full FTA processing pipeline.

    Steps:
        1. Clean dollar columns
        2. Clean string pct columns
        3. Validate float pct columns
        4. Inflate to 2023 dollars
        5. Rename to snake_case
        6. Save to CSV

    Params:
        df: raw FTA DataFrame from load_raw_fta().
        cpi: CPI DataFrame from load_raw_cpi().
    Returns: processed DataFrame with 2023-adjusted cost columns.
    """
    df = _clean_dollar_cols(df)
    df = _clean_pct_cols_str(df)
    df = _validate_pct_cols_float(df)

    multiplier = _get_inflation_multiplier(cpi)
    df = _inflate_to_2023(df, multiplier)

    df = _rename_to_snake(df)

    out_path = DATA_PROCESSED / "fta_processed.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} rows)")

    return df


# Quick-run

if __name__ == "__main__":
    from src.load_fta import load_raw_fta
    from src.load_global_rail import load_raw_cpi

    raw_df = load_raw_fta()
    cpi = load_raw_cpi()
    cpi.columns = ["Year", "CPI_value", "unused", "Index_2021", "Index_2023"]
    cpi = cpi[pd.to_numeric(cpi["Year"], errors="coerce").notna()].copy()
    cpi["Year"] = cpi["Year"].astype(int)
    cpi = cpi[["Year", "Index_2023"]].set_index("Year")

    process_fta(raw_df, cpi)