import pandas as pd
from src.config import FTA_SUMMARY_XL


def load_raw_fta() -> pd.DataFrame:
    """
    Loads the FTA cost summary sheet without any modifications.

    Returns: Raw FTA cost breakdown data (49 rows) as DataFrame.
    Raises: FileNotFoundError if Excel file is missing.
            ValueError if sheet is not found.
    """
    if not FTA_SUMMARY_XL.exists():
        raise FileNotFoundError(
            f"Data file not found: {FTA_SUMMARY_XL}"
        )

    try:
        return pd.read_excel(FTA_SUMMARY_XL, sheet_name="Sheet1", header=0)
    except Exception:
        raise ValueError(
            f"Sheet 'Sheet1' not found in {FTA_SUMMARY_XL.name}"
        )