"""
Bulk insert version of ingest.py
Only changes the upsert_properties function to use bulk inserts in batches of 100
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Database connection - use st.secrets for deployment compatibility
try:
    DB_USER = st.secrets['DB_USER']
    DB_PASSWORD = st.secrets['DB_PASSWORD']
    DB_HOST = st.secrets['DB_HOST']
    DB_NAME = st.secrets['DB_NAME']
    DB_PORT = st.secrets['DB_PORT']
except:
    # Fallback to os.getenv for local development
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_NAME = os.getenv('DB_NAME')
    DB_PORT = os.getenv('DB_PORT', '5432')

ENGINE = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Column mappings and types (same as original)
REQUIRED_COLUMNS_PROPERTIES = [
    "mls_id", "owner_name", "address", "city", "state", "zip_code", "district",
    "bedrooms", "bathrooms", "living_area", "piccount", "list_price", "status", "report_date",
    "buyer_agent", "equity", "tmk_apn", "phone", "email", "last_sale_date",
    "last_sale_price", "year_built", "total_assessed_value"
]

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
    "email": "text",
    "last_sale_date": "date",
    "last_sale_price": "float",
    "year_built": "int",
    "total_assessed_value": "float"
}

REQUIRED_COLUMNS_PERMITS = [
    "case_number", "permit_type", "status", "project_name", "issued_date", "applied_date",
    "expiration_date", "finalized_date", "module_name", "address", "parcel_number", "description"
]

PERMIT_COLUMN_TYPES = {
    "issued_date": "date",
    "applied_date": "date",
    "expiration_date": "date",
    "finalized_date": "date"
}

REQUIRED_COLUMNS_LEGAL = [
    "case_number", "filing_date", "case_type", "status", "party_name", "address", "tmk_apn"
]

LEGAL_COLUMN_TYPES = {
    "filing_date": "date"
}

def clean_df_for_db(df, required_cols, not_null_cols, column_types):
    """Clean dataframe for database insertion"""
    df = df.copy()
    
    # Remove unwanted columns that might come from Excel files
    unwanted_cols = ['Unnamed: 62', 'CITY', 'STATE', 'Zip', 'Unnamed: 0']
    for col in unwanted_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Add missing columns
    for col in required_cols:
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
            df[col] = df[col].replace({float('nan'): None, np.nan: None, 'nan': None, 'NaN': None, 'nan nan': None})

    # Fill NOT NULL text columns with "UNKNOWN"
    for col in not_null_cols:
        if col in df.columns:
            if column_types.get(col, "text") == "text":
                df[col] = df[col].apply(lambda x: x if x is not None else "UNKNOWN")

    return df[required_cols]  # ensure column order

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
    
    print(f"DEBUG: BEFORE cleaning - df shape: {df.shape}")
    print(f"DEBUG: BEFORE cleaning - sample data: {df.head(2).to_dict()}")
    
    df = clean_df_for_db(df, REQUIRED_COLUMNS_PROPERTIES, not_null_cols, PROPERTY_COLUMN_TYPES)
    
    # Add file_type to the dataframe
    df['file_type'] = file_type
    print(f"DEBUG: AFTER cleaning, dataframe shape: {df.shape}")
    print(f"DEBUG: AFTER cleaning - sample data: {df.head(2).to_dict()}")
    print(f"DEBUG: Sample tmk_apn values: {df.get('tmk_apn', pd.Series()).head(3).tolist()}")

    # Process data in batches of 50
    batch_size = 50
    total_rows = len(df)
    
    print(f"DEBUG: Processing all {total_rows} rows in batches of {batch_size}")
    
    added = 0
    updated = 0
    skipped = 0
    changes = []

    for batch_start in range(0, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        batch_df = df.iloc[batch_start:batch_end]
        
        print(f"DEBUG: Processing batch {batch_start//batch_size + 1}: rows {batch_start+1}-{batch_end}")
        
        # Convert batch to list of dictionaries and clean NaN values
        batch_data = []
        for _, row in batch_df.iterrows():
            row_data = row.to_dict()
            # Final cleanup: convert any remaining NaN values to None for database compatibility
            for key, value in row_data.items():
                if pd.isna(value):
                    row_data[key] = None
            batch_data.append(row_data)
        
        # Process each batch in its own transaction
        with ENGINE.begin() as conn:
            # Bulk insert the batch
            if file_type in ["mls", "batch_leads"]:
                # Use mls_id for MLS and Batch Leads data
                print(f"DEBUG: Bulk inserting {len(batch_data)} rows for {file_type} data with ON CONFLICT (mls_id)")
                sql = text("""
                INSERT INTO properties
                (mls_id, owner_name, address, city, state, zip_code, district,
                bedrooms, bathrooms, living_area, piccount, list_price, status, report_date,
                buyer_agent, equity, tmk_apn, phone, email, file_type, last_sale_date,
                last_sale_price, year_built, total_assessed_value)
                VALUES
                (:mls_id, :owner_name, :address, :city, :state, :zip_code, :district,
                :bedrooms, :bathrooms, :living_area, :piccount, :list_price, :status, :report_date,
                :buyer_agent, :equity, :tmk_apn, :phone, :email, :file_type, :last_sale_date,
                :last_sale_price, :year_built, :total_assessed_value)
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
                    file_type = EXCLUDED.file_type,
                last_sale_date = EXCLUDED.last_sale_date,
                last_sale_price = EXCLUDED.last_sale_price,
                year_built = EXCLUDED.year_built,
                total_assessed_value = EXCLUDED.total_assessed_value
                """)
            else:
                # Use tmk_apn for other data types (RPT, eCourt)
                print(f"DEBUG: Bulk inserting {len(batch_data)} rows for non-MLS/Batch data with ON CONFLICT (tmk_apn, report_date)")
                sql = text("""
                INSERT INTO properties
                (mls_id, owner_name, address, city, state, zip_code, district,
                bedrooms, bathrooms, living_area, piccount, list_price, status, report_date,
                buyer_agent, equity, tmk_apn, phone, email, file_type, last_sale_date,
                last_sale_price, year_built, total_assessed_value)
                VALUES
                (:mls_id, :owner_name, :address, :city, :state, :zip_code, :district,
                :bedrooms, :bathrooms, :living_area, :piccount, :list_price, :status, :report_date,
                :buyer_agent, :equity, :tmk_apn, :phone, :email, :file_type, :last_sale_date,
                :last_sale_price, :year_built, :total_assessed_value)
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
                    report_date = EXCLUDED.report_date,
                    buyer_agent = EXCLUDED.buyer_agent,
                    equity = EXCLUDED.equity,
                    phone = EXCLUDED.phone,
                    email = EXCLUDED.email,
                    file_type = EXCLUDED.file_type,
                last_sale_date = EXCLUDED.last_sale_date,
                last_sale_price = EXCLUDED.last_sale_price,
                year_built = EXCLUDED.year_built,
                total_assessed_value = EXCLUDED.total_assessed_value
                """)
            
            try:
                result = conn.execute(sql, batch_data)
                print(f"✅ Successfully inserted batch {batch_start//batch_size + 1}: {len(batch_data)} rows")
                print(f"DEBUG: SQL execution result: {result.rowcount if hasattr(result, 'rowcount') else 'N/A'}")
                print(f"DEBUG: Sample batch data: {batch_data[0] if batch_data else 'Empty batch'}")
                added += len(batch_data)  # Simplified - we'll count all as added for now
            except Exception as e:
                print(f"❌ ERROR: Batch insert failed for batch {batch_start//batch_size + 1}")
                print(f"ERROR: Exception: {e}")
                print(f"DEBUG: Failed batch data sample: {batch_data[0] if batch_data else 'Empty batch'}")
                raise

            # DEBUG: Check actual database count after insertion
            try:
                count_result = conn.execute(text("SELECT COUNT(*) FROM properties WHERE file_type = :file_type"), {"file_type": file_type})
                db_count = count_result.scalar()
                print(f"DEBUG: Total rows in database for file_type '{file_type}': {db_count}")
            except Exception as e:
                print(f"DEBUG: Could not check database count: {e}")
        
        # Transaction automatically commits here due to context manager
        print(f"✅ Batch {batch_start//batch_size + 1} committed successfully")

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
