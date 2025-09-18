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
    "buyer_agent", "equity", "tmk_apn", "phone", "email"
]

PROPERTY_COLUMN_TYPES = {
    "mls_id": "str",
    "owner_name": "str", 
    "address": "str",
    "city": "str",
    "state": "str",
    "zip_code": "str",
    "district": "str",
    "bedrooms": "int",
    "bathrooms": "float",
    "living_area": "int",
    "piccount": "int",
    "list_price": "float",
    "status": "str",
    "report_date": "date",
    "buyer_agent": "str",
    "equity": "float",
    "tmk_apn": "str",
    "phone": "str",
    "email": "str"
}

def clean_df_for_db(df, required_cols, not_null_cols, column_types):
    """Clean dataframe for database insertion"""
    # Ensure all required columns exist
    for col in required_cols:
        if col not in df.columns:
            if col in column_types:
                if column_types[col] == "str":
                    df[col] = ""
                elif column_types[col] == "int":
                    df[col] = 0
                elif column_types[col] == "float":
                    df[col] = 0.0
                elif column_types[col] == "date":
                    df[col] = datetime.now().date()
            else:
                df[col] = None
    
    # Convert data types
    for col, dtype in column_types.items():
        if col in df.columns:
            if dtype == "str":
                df[col] = df[col].astype(str).replace('nan', '')
            elif dtype == "int":
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            elif dtype == "float":
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            elif dtype == "date":
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.date.fillna(datetime.now().date())
    
    # Handle NaN values for not null columns
    for col in not_null_cols:
        if col in df.columns:
            if column_types.get(col) == "str":
                df[col] = df[col].fillna("")
            elif column_types.get(col) == "int":
                df[col] = df[col].fillna(0)
            elif column_types.get(col) == "float":
                df[col] = df[col].fillna(0.0)
            elif column_types.get(col) == "date":
                df[col] = df[col].fillna(datetime.now().date())
    
    return df

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

    # Process data in batches of 100
    batch_size = 100
    total_rows = len(df)
    
    print(f"DEBUG: Processing {total_rows} rows in batches of {batch_size}")
    
    added = 0
    updated = 0
    skipped = 0
    changes = []

    with ENGINE.begin() as conn:
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
            
            # Bulk insert the batch
            if file_type in ["mls", "batch_leads"]:
                # Use mls_id for MLS and Batch Leads data
                print(f"DEBUG: Bulk inserting {len(batch_data)} rows for {file_type} data with ON CONFLICT (mls_id)")
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
                print(f"DEBUG: Bulk inserting {len(batch_data)} rows for non-MLS/Batch data with ON CONFLICT (tmk_apn, report_date)")
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
                    report_date = EXCLUDED.report_date,
                    buyer_agent = EXCLUDED.buyer_agent,
                    equity = EXCLUDED.equity,
                    phone = EXCLUDED.phone,
                    email = EXCLUDED.email,
                    file_type = EXCLUDED.file_type
                """)
            
            try:
                conn.execute(sql, batch_data)
                print(f"✅ Successfully inserted batch {batch_start//batch_size + 1}: {len(batch_data)} rows")
                added += len(batch_data)  # Simplified - we'll count all as added for now
            except Exception as e:
                print(f"❌ ERROR: Batch insert failed for batch {batch_start//batch_size + 1}")
                print(f"ERROR: Exception: {e}")
                raise

    return {
        "added": added, 
        "updated": updated, 
        "skipped": skipped,
        "changes": changes
    }
