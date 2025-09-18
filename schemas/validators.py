# augur_app/schemas/validators.py
import re
import pandas as pd
from typing import Tuple, List, Dict, Any

# Candidate header names for common columns
_CANDIDATES = {
    "mls_id": ["MLS #", "MLS", "MLS#", "mls #", "mls"],
    "report_date": ["Report Date", "report_date", "report date", "Date"],
    "pic_count": ["PicCount", "Pic Count", "Pics"],
    "status": ["Status", "STAT"],
    "type": ["Type"],
    "l_price": ["L Price", "List Price", "Price", "L_Price", "ListPrice", "LPrice"],
    "address": ["Address", "Property Address", "ADDR"],
    "district": ["District"],
    "bds": ["Bds", "Beds", "Bedrooms"],
    "bths": ["Bths", "Baths", "Bathrooms"],
    "liv_sf": ["Liv-SF", "Liv_SF", "Liv SF", "LivSqFt", "Liv_SqFt", "LivSqFt"],
    "city": ["City", "CITY"],
    "state": ["State", "STATE"],
    "zip": ["Zip", "ZIP", "Postal"],
}

def _normalize_colname(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"[^\w]", "", s)  # remove spaces, punctuation
    return s

def _build_lookup_map(columns: List[str]) -> Dict[str, str]:
    """
    Given original dataframe columns, find matching standard names.
    Returns a mapping original_col -> standard_name
    """
    norm_cols = {col: _normalize_colname(col) for col in columns}
    lookup = {}
    # For each standard name, find a matching original column
    for std, candidates in _CANDIDATES.items():
        for cand in candidates:
            cand_norm = _normalize_colname(cand)
            for orig, orig_norm in norm_cols.items():
                if orig_norm == cand_norm:
                    lookup[orig] = std
                    break
            if any(orig for orig in lookup if lookup[orig] == std):
                break
    # As fallback: try matching by substring (e.g., 'reportdate' vs 'report_date')
    for orig, orig_norm in norm_cols.items():
        if orig in lookup:
            continue
        for std, candidates in _CANDIDATES.items():
            for cand in candidates:
                if cand.lower().replace(" ", "") in orig_norm or orig_norm in cand.lower().replace(" ", ""):
                    lookup[orig] = std
                    break
            if any(orig for orig in lookup if lookup[orig] == std):
                break
    return lookup

def _parse_price(x: Any) -> Any:
    if pd.isna(x):
        return None
    s = str(x)
    s = re.sub(r"[^\d.]", "", s)
    if s == "":
        return None
    try:
        # integer cents not needed: convert to int dollars
        val = float(s)
        return int(round(val))
    except:
        return None

def normalize_mls_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Normalize MLS dataframe columns to a standard schema and perform lightweight checks.
    Returns (normalized_df, errors_list).
    """
    df = df.copy()
    # ensure string columns to avoid weird dtype issues
    df.columns = [str(c) for c in df.columns]
    lookup = _build_lookup_map(list(df.columns))
    # rename df columns according to lookup
    rename_map = {orig: std for orig, std in lookup.items()}
    df = df.rename(columns=rename_map)

    # Also handle some common header cases (upper-case CITY / STATE)
    df.columns = [c.strip() for c in df.columns]

    # Parse numeric fields if present
    if "l_price" in df.columns:
        df["l_price"] = df["l_price"].apply(_parse_price)
    if "bds" in df.columns:
        df["bds"] = pd.to_numeric(df["bds"].astype(str).str.replace(",", ""), errors="coerce")
    if "bths" in df.columns:
        df["bths"] = pd.to_numeric(df["bths"].astype(str).str.replace(",", ""), errors="coerce")
    if "liv_sf" in df.columns:
        df["liv_sf"] = pd.to_numeric(df["liv_sf"].astype(str).str.replace(",", ""), errors="coerce")
    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce", infer_datetime_format=True)

    # Minimal validation - require at least address or mls_id, and require status
    errors = []
    if "status" not in df.columns:
        errors.append("Missing required column: 'status'")
    if not ("mls_id" in df.columns or "address" in df.columns):
        errors.append("Missing required identifier column: 'MLS #' or 'Address'")

    # If some rows have completely empty required cells, record that
    if "status" in df.columns:
        empty_status_count = int(df["status"].isna().sum())
        if empty_status_count > 0:
            errors.append(f"{empty_status_count} rows have empty 'status' values")
    if "l_price" in df.columns:
        invalid_prices = int(df["l_price"].isna().sum())
        # don't fail for all missing prices, but warn
        if invalid_prices > 0:
            errors.append(f"{invalid_prices} rows have unparsable 'L Price'")

    # Success: return df and any warnings/errors
    return df, errors
