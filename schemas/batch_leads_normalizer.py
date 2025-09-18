"""
Normalizer for Batch Leads Excel files
Maps Batch Leads data to our database schema
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re

def normalize_batch_leads_df(df):
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
        'Property Zip': 'property_zip',
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
        df_norm['owner_name'] = df_norm['first_name'].astype(str) + ' ' + df_norm['last_name'].astype(str)
        df_norm['owner_name'] = df_norm['owner_name'].replace('nan nan', None)
    
    # Set file_type
    df_norm['file_type'] = 'batch_leads'
    
    # Set report_date to current date if not available
    if 'created_date' in df_norm.columns:
        df_norm['report_date'] = pd.to_datetime(df_norm['created_date'], errors='coerce').dt.date
    else:
        df_norm['report_date'] = datetime.now().date()
    
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
    
    # Clean up any remaining NaN values
    df_norm = df_norm.replace({np.nan: None, 'nan': None, 'NaN': None})
    
    messages.append(f"Normalized {len(df_norm)} Batch Leads records")
    messages.append(f"Key fields: {len([col for col in df_norm.columns if col in ['equity', 'tmk_apn', 'year_built', 'living_area']])} available")
    
    return df_norm, messages

