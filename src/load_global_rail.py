import pandas as pd
from src.config import GLOBAL_RAIL_XL, SHEET_LATEST, SHEET_CPI


def load_raw_projects() -> pd.DataFrame:
    """
    Loads the latest project sheet without any modifications.

    Returns: Raw project data as DataFrame.
    Raises: FileNotFoundError if Excel file is missing.
            ValueError if sheet is not found.
    """
    if not GLOBAL_RAIL_XL.exists():
        raise FileNotFoundError(
            f"Data file not found: {GLOBAL_RAIL_XL}"
        )

    try:
        return pd.read_excel(GLOBAL_RAIL_XL, sheet_name=SHEET_LATEST, header=0)
    except Exception:
        raise ValueError(
            f"Sheet '{SHEET_LATEST}' not found in {GLOBAL_RAIL_XL.name}"
        )


def load_raw_cpi() -> pd.DataFrame:
    """
    Loads the CPI sheet without any modifications.

    Returns: Raw CPI data (1965-2030) as DataFrame.
    Raises: FileNotFoundError if Excel file is missing.
            ValueError if sheet is not found.
    """
    if not GLOBAL_RAIL_XL.exists():
        raise FileNotFoundError(
            f"Data file not found: {GLOBAL_RAIL_XL}"
        )

    try:
        return pd.read_excel(GLOBAL_RAIL_XL, sheet_name=SHEET_CPI, header=0)
    except Exception:
        raise ValueError(
            f"Sheet '{SHEET_CPI}' not found in {GLOBAL_RAIL_XL.name}"
        )