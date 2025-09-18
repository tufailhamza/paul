# ingest/ingest.py

import os
import pandas as pd
from sqlalchemy import create_engine, text
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Database connection - use st.secrets for deployment compatibility
try:
    DB_USER = st.secrets["DB_USER"]
    DB_PASSWORD = st.secrets["DB_PASSWORD"]
    DB_HOST = st.secrets["DB_HOST"]
    DB_NAME = st.secrets["DB_NAME"]
    DB_PORT = st.secrets["DB_PORT"]
except:
    # Fallback to os.getenv for local development
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_NAME = os.getenv("DB_NAME")
    DB_PORT = os.getenv("DB_PORT", "5432")

ENGINE = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# -------------------------
# Required columns per table
# -------------------------
REQUIRED_COLUMNS_PROPERTIES = [
    "mls_id", "owner_name", "address", "city", "state", "zip_code", "district",
    "bedrooms", "bathrooms", "living_area", "piccount", "list_price", "status", "report_date",
    "buyer_agent", "equity", "tmk_apn", "phone", "email"
]

REQUIRED_COLUMNS_PERMITS = [
    "case_number", "permit_type", "status", "project_name", "issued_date", "applied_date",
    "expiration_date", "finalized_date", "module_name", "address", "parcel_number", "description"
]

REQUIRED_COLUMNS_LEGAL = [
    "case_number", "filing_date", "case_type", "status", "party_name", "address", "tmk_apn"
]

# -------------------------
# Helper: clean dataframe
# -------------------------
def clean_df_for_db(df: pd.DataFrame, required_columns: list, not_null_columns: list = [], column_types: dict = {}) -> pd.DataFrame:
    """
    - Converts NaN / NaT → None
    - Converts datetime columns to Python datetime
    - Ensures all required columns exist
    - Fills NOT NULL text columns with "UNKNOWN"
    - Converts numeric/date columns to None if invalid
    """
    import numpy as np
    
    df = df.copy()

    # Add missing columns
    for col in required_columns:
        if col not in df.columns:
            df[col] = None

    # Fill NaN / NaT with None
    for col in df.columns:
        dtype = column_types.get(col, None)
        if pd.api.types.is_datetime64_any_dtype(df[col]) or dtype == "date":
            # Convert invalid dates to None
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = df[col].apply(lambda x: x.to_pydatetime() if pd.notna(x) else None)
        elif dtype in ("int", "float"):
            # Convert non-numeric to None, but preserve existing None values
            df[col] = df[col].apply(lambda x: pd.to_numeric(x, errors="coerce") if x is not None else None)
            # Convert all NaN values to None
            df[col] = df[col].where(pd.notna(df[col]), None)
            # Handle special cases for integer columns
            if dtype == "int":
                # Ensure integer columns are proper integers or None
                df[col] = df[col].apply(lambda x: int(x) if x is not None and pd.notna(x) and not pd.isna(x) and str(x) != 'nan' else None)
        else:
            # Text column: replace NaN with None
            df[col] = df[col].where(pd.notna(df[col]), None)
            # Convert any remaining NaN strings to None
            df[col] = df[col].replace({float('nan'): None, np.nan: None, 'nan': None, 'NaN': None})

    # Fill NOT NULL text columns with "UNKNOWN"
    for col in not_null_columns:
        if col in df.columns:
            if column_types.get(col, "text") == "text":
                df[col] = df[col].apply(lambda x: x if x is not None else "UNKNOWN")

    return df[required_columns]  # ensure column order

# -------------------------
# Column types for tables
# -------------------------
PROPERTY_COLUMN_TYPES = {
    "mls_id": "text",
    "owner_name": "text",
    "address": "text",
    "city": "text",
    "state": "text",
    "zip_code": "text",
    "district": "text",
    "bedrooms": "int",
    "bathrooms": "float",
    "living_area": "int",
    "piccount": "int",
    "list_price": "float",
    "status": "text",
    "report_date": "date",
    "buyer_agent": "text",
    "equity": "float",
    "tmk_apn": "text",
    "phone": "text",
    "email": "text"
}

PERMIT_COLUMN_TYPES = {
    "issued_date": "date",
    "applied_date": "date",
    "expiration_date": "date",
    "finalized_date": "date"
}

LEGAL_COLUMN_TYPES = {
    "filing_date": "date"
}

# -------------------------
# Upsert functions
# -------------------------

def upsert_properties(df: pd.DataFrame, file_type: str = "mls"):
    print(f"DEBUG: Starting upsert_properties for file_type: {file_type}")
    print(f"DEBUG: Input dataframe shape: {df.shape}")
    print(f"DEBUG: Sample mls_id values: {df.get('mls_id', pd.Series()).head(3).tolist()}")
    
    # For MLS and Batch Leads data, use mls_id for conflict resolution to avoid unique constraint violations
    if file_type in ["mls", "batch_leads"]:
        not_null_cols = ["address", "mls_id", "report_date"]
        print(f"DEBUG: Using {file_type}-specific NOT NULL columns (excluding tmk_apn)")
    else:
        not_null_cols = ["address", "mls_id", "tmk_apn", "report_date"]
        print("DEBUG: Using standard NOT NULL columns (including tmk_apn)")
    
    df = clean_df_for_db(df, REQUIRED_COLUMNS_PROPERTIES, not_null_cols, PROPERTY_COLUMN_TYPES)
    
    # Add file_type to the dataframe
    df['file_type'] = file_type
    print(f"DEBUG: After cleaning, dataframe shape: {df.shape}")
    print(f"DEBUG: Sample tmk_apn values: {df.get('tmk_apn', pd.Series()).head(3).tolist()}")

    added = 0
    updated = 0
    skipped = 0
    changes = []

    with ENGINE.begin() as conn:
        for _, row in df.iterrows():
            row_data = row.to_dict()
            
            # Final cleanup: convert any remaining NaN values to None for database compatibility
            for key, value in row_data.items():
                if pd.isna(value):
                    row_data[key] = None
            
            # Check if record exists - use mls_id for MLS data, tmk_apn for other data
            mls_id = row_data.get('mls_id', '')
            is_fallback = mls_id.startswith('FALLBACK_')
            print(f"DEBUG: Processing row with mls_id: {mls_id}, is_fallback: {is_fallback}")
            
            if file_type in ["mls", "batch_leads"]:
                # Use mls_id for MLS and Batch Leads data
                print(f"DEBUG: Using mls_id for conflict resolution ({file_type} data)")
                check_sql = text("""
                SELECT id, owner_name, address, status, list_price, equity
                FROM properties 
                WHERE mls_id = :mls_id
                """)
                existing = conn.execute(check_sql, {
                    'mls_id': row_data['mls_id']
                }).fetchone()
            else:
                # Use tmk_apn for other data types (RPT, eCourt)
                print("DEBUG: Using tmk_apn for conflict resolution (non-MLS/Batch data)")
                check_sql = text("""
                SELECT id, owner_name, address, status, list_price, equity
                FROM properties 
                WHERE tmk_apn = :tmk_apn AND report_date = :report_date
                """)
                existing = conn.execute(check_sql, {
                    'tmk_apn': row_data['tmk_apn'], 
                    'report_date': row_data['report_date']
                }).fetchone()
            
            if existing:
                # Check for changes
                changes_detected = []
                if existing.owner_name != row_data['owner_name']:
                    changes_detected.append(f"owner_name: {existing.owner_name} → {row_data['owner_name']}")
                if existing.address != row_data['address']:
                    changes_detected.append(f"address: {existing.address} → {row_data['address']}")
                if existing.status != row_data['status']:
                    changes_detected.append(f"status: {existing.status} → {row_data['status']}")
                if existing.list_price != row_data['list_price']:
                    changes_detected.append(f"price: {existing.list_price} → {row_data['list_price']}")
                if existing.equity != row_data['equity']:
                    changes_detected.append(f"equity: {existing.equity} → {row_data['equity']}")
                
                if changes_detected:
                    changes.append({
                        'tmk_apn': row_data['tmk_apn'],
                        'address': row_data['address'],
                        'changes': changes_detected
                    })
                    updated += 1
                else:
                    skipped += 1
            else:
                added += 1
            
            # Perform upsert - use different conflict resolution based on data type
            if file_type in ["mls", "batch_leads"]:
                # Use mls_id for MLS and Batch Leads data
                print(f"DEBUG: Executing SQL for {file_type} data with ON CONFLICT (mls_id)")
                sql = text("""
                INSERT INTO properties
                (mls_id, owner_name, address, city, state, zip_code, district,
                bedrooms, bathrooms, living_area, piccount, list_price, status, report_date,
                buyer_agent, equity, tmk_apn, phone, email, file_type)
                VALUES
                (:mls_id, :owner_name, :address, :city, :state, :zip_code, :district,
                :bedrooms, :bathrooms, :living_area, :piccount, :list_price, :status, :report_date,
                :buyer_agent, :equity, :tmk_apn, :phone, :email, :file_type)
                ON CONFLICT (mls_id)
                DO UPDATE SET
                    owner_name = EXCLUDED.owner_name,
                    address = EXCLUDED.address,
                    city = EXCLUDED.city,
                    state = EXCLUDED.state,
                    zip_code = EXCLUDED.zip_code,
                    district = EXCLUDED.district,
                    bedrooms = EXCLUDED.bedrooms,
                    bathrooms = EXCLUDED.bathrooms,
                    living_area = EXCLUDED.living_area,
                    piccount = EXCLUDED.piccount,
                    list_price = EXCLUDED.list_price,
                    status = EXCLUDED.status,
                    report_date = EXCLUDED.report_date,
                    buyer_agent = EXCLUDED.buyer_agent,
                    equity = EXCLUDED.equity,
                    tmk_apn = EXCLUDED.tmk_apn,
                    phone = EXCLUDED.phone,
                    email = EXCLUDED.email,
                    file_type = EXCLUDED.file_type
                """)
            else:
                # Use tmk_apn for other data types (RPT, eCourt)
                print("DEBUG: Executing SQL for non-MLS/Batch data with ON CONFLICT (tmk_apn, report_date)")
                sql = text("""
                INSERT INTO properties
                (mls_id, owner_name, address, city, state, zip_code, district,
                bedrooms, bathrooms, living_area, piccount, list_price, status, report_date,
                buyer_agent, equity, tmk_apn, phone, email, file_type)
                VALUES
                (:mls_id, :owner_name, :address, :city, :state, :zip_code, :district,
                :bedrooms, :bathrooms, :living_area, :piccount, :list_price, :status, :report_date,
                :buyer_agent, :equity, :tmk_apn, :phone, :email, :file_type)
                ON CONFLICT (tmk_apn, report_date)
                DO UPDATE SET
                    owner_name = EXCLUDED.owner_name,
                    address = EXCLUDED.address,
                    city = EXCLUDED.city,
                    state = EXCLUDED.state,
                    zip_code = EXCLUDED.zip_code,
                    district = EXCLUDED.district,
                    bedrooms = EXCLUDED.bedrooms,
                    bathrooms = EXCLUDED.bathrooms,
                    living_area = EXCLUDED.living_area,
                    piccount = EXCLUDED.piccount,
                    list_price = EXCLUDED.list_price,
                    status = EXCLUDED.status,
                    buyer_agent = EXCLUDED.buyer_agent,
                    equity = EXCLUDED.equity,
                    phone = EXCLUDED.phone,
                    email = EXCLUDED.email,
                    file_type = EXCLUDED.file_type
                """)
            try:
                print(f"DEBUG: Executing SQL with parameters: mls_id={row_data.get('mls_id')}, tmk_apn={row_data.get('tmk_apn')}, report_date={row_data.get('report_date')}")
                conn.execute(sql, row_data)
                print(f"DEBUG: SQL executed successfully for mls_id: {row_data.get('mls_id')}")
            except Exception as e:
                print(f"ERROR: SQL execution failed for mls_id: {row_data.get('mls_id')}")
                print(f"ERROR: SQL parameters: {row_data}")
                print(f"ERROR: Exception: {e}")
                raise

    return {
        "added": added, 
        "updated": updated, 
        "skipped": skipped,
        "changes": changes
    }

def upsert_permits(df: pd.DataFrame):
    df = clean_df_for_db(df, REQUIRED_COLUMNS_PERMITS, [], PERMIT_COLUMN_TYPES)
    added = 0
    updated = 0

    with ENGINE.begin() as conn:
        for _, row in df.iterrows():
            row_data = row.to_dict()
            sql = text("""
            INSERT INTO permits
            (case_number, permit_type, status, project_name, issued_date, applied_date,
            expiration_date, finalized_date, module_name, address, parcel_number, description)
            VALUES
            (:case_number, :permit_type, :status, :project_name, :issued_date, :applied_date,
            :expiration_date, :finalized_date, :module_name, :address, :parcel_number, :description)
            ON CONFLICT (case_number)
            DO UPDATE SET
                permit_type = EXCLUDED.permit_type,
                status = EXCLUDED.status,
                project_name = EXCLUDED.project_name,
                issued_date = EXCLUDED.issued_date,
                applied_date = EXCLUDED.applied_date,
                expiration_date = EXCLUDED.expiration_date,
                finalized_date = EXCLUDED.finalized_date,
                module_name = EXCLUDED.module_name,
                address = EXCLUDED.address,
                parcel_number = EXCLUDED.parcel_number,
                description = EXCLUDED.description
            RETURNING id;
            """)
            result = conn.execute(sql, row_data)
            if result.rowcount == 1:
                added += 1
            else:
                updated += 1

    return {"added": added, "updated": updated}

def upsert_legal_events(df: pd.DataFrame):
    df = clean_df_for_db(df, REQUIRED_COLUMNS_LEGAL, [], LEGAL_COLUMN_TYPES)
    added = 0
    updated = 0

    with ENGINE.begin() as conn:
        for _, row in df.iterrows():
            row_data = row.to_dict()
            sql = text("""
            INSERT INTO legal_events
            (case_number, filing_date, case_type, status, party_name, address, tmk_apn)
            VALUES
            (:case_number, :filing_date, :case_type, :status, :party_name, :address, :tmk_apn)
            ON CONFLICT (case_number)
            DO UPDATE SET
                filing_date = EXCLUDED.filing_date,
                case_type = EXCLUDED.case_type,
                status = EXCLUDED.status,
                party_name = EXCLUDED.party_name,
                address = EXCLUDED.address,
                tmk_apn = EXCLUDED.tmk_apn
            RETURNING id;
            """)
            result = conn.execute(sql, row_data)
            if result.rowcount == 1:
                added += 1
            else:
                updated += 1

    return {"added": added, "updated": updated}
