# augur_app/schemas/normalizers.py
"""
Normalizers + lightweight validators for MLS, MAPPS, Batch Leads, eCourt.
Fixes:
 - Deduplicate columns automatically
 - Consistent ERROR handling for required fields
 - Clean normalization across all 4 file types
"""

from typing import Tuple, List
import pandas as pd
import re
from datetime import datetime

# ========== helpers ==========

def _normalize_colname(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"[^\w]", "_", s)  # non-alnum -> underscore
    s = re.sub(r"_+", "_", s)
    s = s.strip("_").lower()
    return s


def _dedup_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Drop duplicate columns.
       - If duplicates have identical values → drop silently.
       - If duplicates differ → keep first, drop others, and add warning.
       Returns (df, messages).
    """
    messages = []
    drop = []

    print("DEBUG: Original columns ->", list(df.columns))

    for col in df.columns.unique():
        dupes = [c for c in df.columns if c == col]
        if len(dupes) > 1:
            print(f"DEBUG: Found duplicates for '{col}' -> {dupes}")
            first = df[dupes[0]]
            all_same = all(df[c].equals(first) for c in dupes[1:])
            if all_same:
                drop.extend(dupes[1:])
                print(f"DEBUG: '{col}' duplicates are IDENTICAL → dropping {dupes[1:]}")
            else:
                drop.extend(dupes[1:])
                msg = f"WARNING: Duplicate column '{col}' had conflicting values → kept first, dropped others"
                print("DEBUG:", msg)
                messages.append(msg)

    if drop:
        print("DEBUG: Dropping columns ->", drop)
        df = df.drop(columns=drop)

    print("DEBUG: Final columns ->", list(df.columns))
    return df, messages

def _build_col_map(cols):
    norm = {c: _normalize_colname(c) for c in cols}
    col_map = {}
    candidates = {
        "mls_id": ["mls", "mls_no", "mls_number", "mls_#", "mlsnum"],
        "piccount": ["piccount", "pics", "pic_count"],
        "status": ["status", "stat"],
        "property_type": ["type", "property_type"],
        "permit_type": ["type", "permit_type"],
        "list_price": ["l_price", "list_price", "price"],
        "address": ["address", "property_address", "addr"],
        "district": ["district", "district_"],
        "bedrooms": ["bds", "beds", "bedrooms"],
        "bathrooms": ["bths", "baths", "bathrooms"],
        "living_area": ["liv_sf", "living_area", "sqft", "liv_sqft"],
        "city": ["city", "city_", "town"],  # Handle "City " with trailing space
        "state": ["state"],
        "zip_code": ["zip", "zip_code", "postal"],
        "parcel_number": ["main_parcel", "parcel_number", "parcel"],
        "agent_name": ["agent_agt_nm_ph", "agent_name", "buyer_agent", "agent"],
        "report_date": ["report_date", "date"],
        "case_number": ["case_number", "case_no", "case_num"],
        "view": ["view"],
        "wtrfrt": ["wtrfrt", "waterfront"],
        "buyer_agent": ["buys_agt_agt_name", "buyer_agent", "buyer_agt"]
    }
    for orig, onorm in norm.items():
        found = False
        # exact match first
        for target, variants in candidates.items():
            if onorm in variants:
                col_map[orig] = target
                found = True
                break
        if not found:
            # fallback: substring match
            for target, variants in candidates.items():
                for v in variants:
                    if v in onorm or onorm in v:
                        col_map[orig] = target
                        found = True
                        break
                if found:
                    break
    return col_map


# --- parsers (unchanged) ---

def _parse_price(val):
    if pd.isna(val): return None
    s = re.sub(r"[^\d.]", "", str(val))
    if not s: return None
    try: return float(s)
    except: return None

def _parse_int(val):
    if pd.isna(val): return None
    try: 
        result = int(float(str(val).replace(",", "").strip()))
        return result if not pd.isna(result) else None
    except: 
        return None

def _parse_float(val):
    if pd.isna(val): return None
    try: 
        result = float(str(val).replace(",", "").strip())
        return result if not pd.isna(result) else None
    except: 
        return None

def _parse_date(val):
    if pd.isna(val): return None
    if isinstance(val, (pd.Timestamp, datetime)):
        return pd.to_datetime(val).date()
    s = str(val).strip()
    if not s: return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d"):
        try: return datetime.strptime(s, fmt).date()
        except: continue
    try:
        return pd.to_datetime(s, errors="coerce").date()
    except: return None

# ========== MLS normalizer ==========
def normalize_mls_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    messages = []

    df, dedup_msgs = _dedup_columns(df)
    messages.extend(dedup_msgs)

    df = df.rename(columns=_build_col_map(df.columns))

    # Required fields (common across all MLS files)
    required_fields = [
        "status", "property_type", "list_price", "address", "city", "state",
        "zip_code", "district", "bedrooms", "bathrooms", "living_area"
    ]
    
    # Optional fields (may be missing in some files)
    optional_fields = [
        "mls_id", "piccount", "report_date", "agent_name", "buyer_agent", 
        "view", "wtrfrt"
    ]
    
    desired = required_fields + optional_fields

    if "list_price" in df: df["list_price"] = df["list_price"].apply(_parse_price)
    if "bedrooms" in df: df["bedrooms"] = df["bedrooms"].apply(_parse_int)
    if "bathrooms" in df: df["bathrooms"] = df["bathrooms"].apply(_parse_float)
    if "living_area" in df: df["living_area"] = df["living_area"].apply(_parse_int)
    if "piccount" in df: 
        df["piccount"] = df["piccount"].apply(_parse_int)
        # Ensure piccount is None instead of NaN for database compatibility
        df["piccount"] = df["piccount"].where(pd.notna(df["piccount"]), None)
        # Convert any remaining NaN values to None
        df["piccount"] = df["piccount"].replace({float('nan'): None})
    if "city" in df: df["city"] = df["city"].astype(str).str.strip().str.upper()
    if "state" in df: df["state"] = df["state"].astype(str).str.strip().str.upper()
    if "zip_code" in df: df["zip_code"] = df["zip_code"].astype(str).str.strip().str[:5]
    if "report_date" in df: 
        df["report_date"] = df["report_date"].apply(_parse_date)
    else:
        # If no report_date, use current date
        df["report_date"] = datetime.now().date()

    # Handle missing mls_id (for Active MLS files)
    if "mls_id" not in df:
        messages.append("WARNING: mls_id missing - generating fallback IDs")
        df["mls_id"] = df.apply(
            lambda r: "FALLBACK_" + str(abs(hash(
                f"{r.get('address','')}|{r.get('zip_code','')}|{r.get('status','')}|{r.get('report_date','')}"
            )))[:15], axis=1)
    
    # Handle missing piccount (for Pending MLS files)
    if "piccount" not in df:
        messages.append("WARNING: piccount missing - setting to 0")
        df["piccount"] = 0

    # Check for truly critical required fields (address is the only one that can't be defaulted)
    critical_required = ["address"]
    missing_critical = [field for field in critical_required if field not in df]
    if missing_critical:
        messages.append(f"ERROR: Missing critical required fields: {missing_critical}")
        return pd.DataFrame(), messages
    
    # Create normalized dataframe with available fields
    df_norm = df[[c for c in desired if c in df]].copy()
    
    # Handle mls_id fallback generation BEFORE adding missing fields
    if "mls_id" not in df_norm:
        messages.append("WARNING: mls_id missing - generating fallback IDs")
        df_norm["mls_id"] = [f"FALLBACK_{i:06d}" for i in range(len(df_norm))]
        print(f"DEBUG: Generated {len(df_norm)} fallback IDs: {df_norm['mls_id'].head(3).tolist()}")
    
    # Add all missing fields (both required and optional) to ensure they're in the final output
    for field in desired:
        if field not in df_norm:
            df_norm[field] = None
    
    # Add missing required fields with defaults
    for field in required_fields:
        if field not in df.columns:
            if field == "status":
                messages.append("WARNING: status missing - setting to 'UNKNOWN'")
                df_norm["status"] = "UNKNOWN"
            elif field == "district":
                messages.append("WARNING: district missing - setting to 'UNKNOWN'")
                df_norm["district"] = "UNKNOWN"
            else:
                df_norm[field] = None
    
    # Add missing optional fields with defaults
    for field in optional_fields:
        if field not in df_norm:
            if field == "piccount":
                messages.append("WARNING: piccount missing - setting to 0")
                df_norm["piccount"] = 0
            elif field == "report_date":
                messages.append("WARNING: report_date missing - using current date")
                df_norm["report_date"] = datetime.now().date()
            else:
                df_norm[field] = None
    
    # Ensure required fields have values
    if "address" in df_norm:
        # Fill missing addresses with a placeholder
        df_norm["address"] = df_norm["address"].fillna("UNKNOWN")
    if "tmk_apn" not in df_norm:
        # Don't set tmk_apn to UNKNOWN for MLS data - let it be NULL
        # This prevents unique constraint violations
        df_norm["tmk_apn"] = None
    if "district" not in df_norm:
        df_norm["district"] = "UNKNOWN"

    return df_norm, messages


# ========== MAPPS normalizer ==========
def normalize_mapps_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    messages = []
    df, dedup_msgs = _dedup_columns(df)
    messages.extend(dedup_msgs)
    df = df.rename(columns=_build_col_map(df.columns))
    
    # Fix specific mapping for MAPPS - "Type" should map to "permit_type", not "property_type"
    if "property_type" in df.columns:
        df = df.rename(columns={"property_type": "permit_type"})
        messages.append("WARNING: Mapped 'Type' column to 'permit_type' for MAPPS data")

    for dt in ["issued_date","applied_date","expiration_date","finalized_date"]:
        if dt in df: df[dt] = df[dt].apply(_parse_date)

    if "address" in df: 
        df["address"] = df["address"].astype(str).str.strip()
        # Convert 'nan' strings to None
        df["address"] = df["address"].replace({'nan': None, 'NaN': None, '': None})
    if "parcel_number" in df: 
        df["parcel_number"] = df["parcel_number"].astype(str).str.strip()
        # Convert 'nan' strings to None
        df["parcel_number"] = df["parcel_number"].replace({'nan': None, 'NaN': None, '': None})
    if "project_name" in df:
        # Convert 'nan' strings to None for project_name
        df["project_name"] = df["project_name"].replace({'nan': None, 'NaN': None, '': None})

    desired = ["case_number","permit_type","status","project_name","issued_date","applied_date",
               "expiration_date","finalized_date","module_name","address","parcel_number","description"]
    df_norm = df[[c for c in desired if c in df]].copy()
    
    if "case_number" not in df_norm:
        messages.append("ERROR: 'case_number' required for MAPPS")
    else:
        # Filter out rows with missing case numbers
        original_count = len(df_norm)
        df_norm = df_norm.dropna(subset=['case_number'])
        df_norm = df_norm[df_norm['case_number'] != '']
        df_norm = df_norm[df_norm['case_number'].str.strip() != '']
        filtered_count = len(df_norm)
        if original_count > filtered_count:
            messages.append(f"WARNING: Filtered out {original_count - filtered_count} rows with missing case numbers")
    
    return df_norm, messages

# ========== Batch Leads normalizer ==========
def normalize_batch_leads_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Normalize Batch Leads Excel data to our database schema
    
    Args:
        df: Raw DataFrame from Batch Leads Excel file
        
    Returns:
        tuple: (normalized_df, messages)
    """
    messages = []
    df_norm = df.copy()
    
    # Column mapping from Batch Leads to our schema
    column_mapping = {
        'First Name': 'first_name',
        'Last Name': 'last_name', 
        'Mailing Address': 'mailing_address',
        'Mailing City': 'mailing_city',
        'Mailing State': 'mailing_state',
        'Mailing Zip': 'mailing_zip',
        'Mailing County': 'mailing_county',
        'Property City': 'city',
        'Property State': 'state',
        'Property Address': 'address',
        'Property Zip': 'zip_code',
        'Email': 'email',
        'Email 2': 'email_2',
        'Phone 1': 'phone_1',
        'Phone 2': 'phone_2',
        'Phone 3': 'phone_3',
        'Phone 4': 'phone_4',
        'Phone 5': 'phone_5',
        'Is Vacant': 'is_vacant',
        'Created Date': 'created_date',
        'Updated Date': 'updated_date',
        'Apn': 'tmk_apn',
        'Property Type Detail': 'property_type_detail',
        'Owner Occupied': 'owner_occupied',
        'Bedroom Count': 'bedrooms',
        'Bathroom Count': 'bathrooms',
        'Total Building Area Square Feet': 'living_area',
        'Lot Size Square Feet': 'lot_size_sqft',
        'Year Built': 'year_built',
        'Total Assessed Value': 'total_assessed_value',
        'Zoning Code': 'zoning_code',
        'Last Sale Date': 'last_sale_date',
        'Last Sale Price': 'last_sale_price',
        'Total Loan Balance': 'total_loan_balance',
        'Equity Current Estimated Balance': 'equity',
        'Estimated Value': 'estimated_value',
        'Ltv Current Estimated Combined': 'ltv_current_combined',
        'Mls Status': 'status',
        'Self Managed': 'self_managed',
        'Loan Recording Date': 'loan_recording_date',
        'Loan Type': 'loan_type',
        'Loan Amount': 'loan_amount',
        'Loan Lender Name': 'loan_lender_name',
        'Loan Due Date': 'loan_due_date',
        'Loan Est Payment': 'loan_est_payment',
        'Loan Est Interest Rate': 'loan_est_interest_rate',
        'Loan Est Balance': 'loan_est_balance',
        'Loan Term (Months)': 'loan_term_months',
        'ARV': 'arv',
        'Spread': 'spread',
        '% ARV': 'pct_arv',
        'Batchrank Score Category': 'batchrank_score_category',
        'Tag Names': 'tag_names',
        'Foreclosure Document Type': 'foreclosure_doc_type',
        'Foreclosure Status': 'foreclosure_status',
        'Foreclosure Auction Date': 'foreclosure_auction_date',
        'Foreclosure Loan Default Date': 'foreclosure_loan_default_date',
        'Foreclosure Recording Date': 'foreclosure_recording_date',
        'Foreclosure Case Number': 'foreclosure_case_number',
        'Foreclosure Trustee/Attorney Name': 'foreclosure_trustee_attorney',
        'Mls Listing Date': 'mls_listing_date',
        'Mls Listing Amount': 'mls_listing_amount'
    }
    
    # Rename columns
    df_norm = df_norm.rename(columns=column_mapping)
    
    # Create owner_name from first_name and last_name
    if 'first_name' in df_norm.columns and 'last_name' in df_norm.columns:
        # Clean first_name and last_name first
        df_norm['first_name'] = df_norm['first_name'].replace({float('nan'): None, 'nan': None, 'NaN': None})
        df_norm['last_name'] = df_norm['last_name'].replace({float('nan'): None, 'nan': None, 'NaN': None})
        
        # Create owner_name
        df_norm['owner_name'] = df_norm.apply(
            lambda row: f"{row['first_name']} {row['last_name']}".strip() 
            if row['first_name'] is not None and row['last_name'] is not None 
            else None, axis=1
        )
    
    # Set file_type
    df_norm['file_type'] = 'batch_leads'
    
    # Set report_date to current date if not available
    if 'created_date' in df_norm.columns:
        df_norm['report_date'] = pd.to_datetime(df_norm['created_date'], errors='coerce').dt.date
    else:
        df_norm['report_date'] = datetime.now().date()
    
    # Clean city and state fields - check if they exist after renaming
    if 'city' in df_norm.columns:
        df_norm['city'] = df_norm['city'].astype(str).str.upper().str.strip()
    
    if 'state' in df_norm.columns:
        df_norm['state'] = df_norm['state'].astype(str).str.upper().str.strip()
    
    # Clean and convert numeric fields
    numeric_fields = [
        'bedrooms', 'bathrooms', 'living_area', 'lot_size_sqft', 'year_built',
        'total_assessed_value', 'zoning_code', 'last_sale_price', 'total_loan_balance',
        'equity', 'estimated_value', 'ltv_current_combined', 'loan_amount',
        'loan_est_payment', 'loan_est_interest_rate', 'loan_est_balance',
        'loan_term_months', 'mls_listing_amount'
    ]
    
    for field in numeric_fields:
        if field in df_norm.columns:
            df_norm[field] = pd.to_numeric(df_norm[field], errors='coerce')
    
    # Clean ARV, Spread, % ARV fields (they might be strings with $ and %)
    for field in ['arv', 'spread', 'pct_arv']:
        if field in df_norm.columns:
            df_norm[field] = df_norm[field].astype(str).str.replace('$', '').str.replace('%', '').str.replace(',', '')
            df_norm[field] = pd.to_numeric(df_norm[field], errors='coerce')
    
    # Convert date fields
    date_fields = [
        'created_date', 'updated_date', 'last_sale_date', 'loan_recording_date',
        'loan_due_date', 'foreclosure_auction_date', 'foreclosure_loan_default_date',
        'foreclosure_recording_date', 'mls_listing_date'
    ]
    
    for field in date_fields:
        if field in df_norm.columns:
            df_norm[field] = pd.to_datetime(df_norm[field], errors='coerce').dt.date
    
    # Clean phone numbers
    phone_fields = ['phone_1', 'phone_2', 'phone_3', 'phone_4', 'phone_5']
    for field in phone_fields:
        if field in df_norm.columns:
            df_norm[field] = df_norm[field].astype(str).str.replace('.0', '')
            df_norm[field] = df_norm[field].replace('nan', None)
    
    # Set default values for required fields
    if 'status' not in df_norm.columns or df_norm['status'].isna().all():
        df_norm['status'] = 'UNKNOWN'
    
    if 'district' not in df_norm.columns:
        df_norm['district'] = 'Kihei'  # Default for Kihei data
    
    # Generate MLS ID if not available (for consistency with other data)
    if 'mls_id' not in df_norm.columns:
        df_norm['mls_id'] = [f"BATCH_{i:06d}" for i in range(len(df_norm))]
    
    # Clean up any remaining NaN values - convert to None for database compatibility
    df_norm = df_norm.replace({float('nan'): None, 'nan': None, 'NaN': None, 'nan nan': None})
    
    # Ensure numeric fields are properly handled for database insertion
    for field in numeric_fields:
        if field in df_norm.columns:
            # Convert NaN to None for database compatibility
            df_norm[field] = df_norm[field].where(pd.notna(df_norm[field]), None)
            # Additional cleanup for any remaining NaN values
            df_norm[field] = df_norm[field].replace({float('nan'): None, 'nan': None, 'NaN': None, 'nan nan': None})
    
    # Handle phone fields specifically
    for field in phone_fields:
        if field in df_norm.columns:
            df_norm[field] = df_norm[field].where(pd.notna(df_norm[field]), None)
            df_norm[field] = df_norm[field].replace({float('nan'): None, 'nan': None, 'NaN': None, 'nan nan': None})
    
    # Ensure tmk_apn is properly handled
    if 'tmk_apn' in df_norm.columns:
        df_norm['tmk_apn'] = df_norm['tmk_apn'].where(pd.notna(df_norm['tmk_apn']), None)
        # Replace empty strings with None
        df_norm['tmk_apn'] = df_norm['tmk_apn'].replace('', None)
        # Keep as None for Batch Leads data to avoid unique constraint issues
    
    # Final cleanup for all text fields
    for col in df_norm.columns:
        if df_norm[col].dtype == 'object':  # Text columns
            df_norm[col] = df_norm[col].replace({float('nan'): None, 'nan': None, 'NaN': None, 'nan nan': None})
            df_norm[col] = df_norm[col].where(pd.notna(df_norm[col]), None)
    
    messages.append(f"Normalized {len(df_norm)} Batch Leads records")
    messages.append(f"Key fields: {len([col for col in df_norm.columns if col in ['equity', 'tmk_apn', 'year_built', 'living_area']])} available")
    
    return df_norm, messages

# ========== eCourt normalizer ==========
def normalize_ecourt_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    messages = []
    df, dedup_msgs = _dedup_columns(df)
    messages.extend(dedup_msgs)
    df = df.rename(columns=_build_col_map(df.columns))
    if "filing_date" in df: df["filing_date"] = df["filing_date"].apply(_parse_date)
    desired = ["case_number","filing_date","case_type","status","party_name","address","tmk_apn"]
    df_norm = df[[c for c in desired if c in df]].copy()
    if "case_number" not in df_norm: messages.append("ERROR: 'case_number' required for eCourt")
    return df_norm, messages

# ========== RPT normalizer ==========
def normalize_rpt_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    messages = []
    df, dedup_msgs = _dedup_columns(df)
    messages.extend(dedup_msgs)
    df = df.rename(columns=_build_col_map(df.columns))

    if "address" not in df:
        if "property_address" in df: df = df.rename(columns={"property_address":"address"})
        else: messages.append("ERROR: address required for RPT")

    if "city" in df: df["city"] = df["city"].astype(str).str.upper().str.strip()
    if "state" in df: df["state"] = df["state"].astype(str).str.upper().str.strip()
    if "zip_code" in df: df["zip_code"] = df["zip_code"].astype(str).str.strip().str[:5]

    desired = ["owner_name","address","city","state","zip_code","tmk_apn","property_type","bedrooms","bathrooms","living_area","year_built","assessed_value"]
    df_norm = df[[c for c in desired if c in df]].copy()

    if "owner_name" not in df_norm: messages.append("ERROR: 'owner_name' required for RPT")
    return df_norm, messages

# ========== Dispatcher ==========
def validate_and_normalize(path_or_df, file_type_hint: str):
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df
    else:
        p = str(path_or_df)
        df = pd.read_csv(p, dtype=str) if p.lower().endswith(".csv") else pd.read_excel(p, dtype=str)

    # Debug: show raw headers
    print(f"[DEBUG] Raw headers: {list(df.columns)}")

    # If duplicate headers exist at load time → dedup immediately
    if df.columns.duplicated().any():
        dupes = [c for c in df.columns[df.columns.duplicated()]]
        print(f"[DEBUG] Found duplicate headers at load: {dupes}")
        # Use your dedup logic here
        df, _ = _dedup_columns(df)

    if file_type_hint.startswith("mls"): 
        return normalize_mls_df(df)
    if file_type_hint.startswith("mapp"): 
        return normalize_mapps_df(df)
    if file_type_hint.startswith("batch") or file_type_hint.startswith("lead"): 
        return normalize_batch_leads_df(df)
    if file_type_hint.startswith("rpt"):
        return normalize_rpt_df(df)
    return normalize_ecourt_df(df)
