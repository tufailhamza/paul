"""
JSON Schema for Batch Leads Excel files
This handles the rich data from Batch Leads Excel files
"""

BATCH_LEADS_SCHEMA = {
    "type": "object",
    "properties": {
        "first_name": {"type": ["string", "null"]},
        "last_name": {"type": ["string", "null"]},
        "mailing_address": {"type": ["string", "null"]},
        "mailing_city": {"type": ["string", "null"]},
        "mailing_state": {"type": ["string", "null"]},
        "mailing_zip": {"type": ["string", "null"]},
        "mailing_county": {"type": ["string", "null"]},
        "property_city": {"type": ["string", "null"]},
        "property_state": {"type": ["string", "null"]},
        "address": {"type": ["string", "null"]},
        "property_zip": {"type": ["string", "null"]},
        "email": {"type": ["string", "null"]},
        "email_2": {"type": ["string", "null"]},
        "phone_1": {"type": ["string", "null"]},
        "phone_2": {"type": ["string", "null"]},
        "phone_3": {"type": ["string", "null"]},
        "phone_4": {"type": ["string", "null"]},
        "phone_5": {"type": ["string", "null"]},
        "is_vacant": {"type": ["string", "null"]},
        "created_date": {"type": ["string", "null", "object"]},
        "updated_date": {"type": ["string", "null", "object"]},
        "apn": {"type": ["string", "null"]},
        "property_type_detail": {"type": ["string", "null"]},
        "owner_occupied": {"type": ["string", "null"]},
        "bedroom_count": {"type": ["number", "null"]},
        "bathroom_count": {"type": ["number", "null"]},
        "total_building_area_square_feet": {"type": ["number", "null"]},
        "lot_size_square_feet": {"type": ["number", "null"]},
        "year_built": {"type": ["number", "null"]},
        "total_assessed_value": {"type": ["number", "null"]},
        "zoning_code": {"type": ["number", "null"]},
        "last_sale_date": {"type": ["string", "null", "object"]},
        "last_sale_price": {"type": ["number", "null"]},
        "total_loan_balance": {"type": ["number", "null"]},
        "equity_current_estimated_balance": {"type": ["number", "null"]},
        "estimated_value": {"type": ["number", "null"]},
        "ltv_current_estimated_combined": {"type": ["number", "null"]},
        "mls_status": {"type": ["string", "null"]},
        "self_managed": {"type": ["string", "null"]},
        "loan_recording_date": {"type": ["string", "null", "object"]},
        "loan_type": {"type": ["string", "null"]},
        "loan_amount": {"type": ["number", "null"]},
        "loan_lender_name": {"type": ["string", "null"]},
        "loan_due_date": {"type": ["string", "null", "object"]},
        "loan_est_payment": {"type": ["number", "null"]},
        "loan_est_interest_rate": {"type": ["number", "null"]},
        "loan_est_balance": {"type": ["number", "null"]},
        "loan_term_months": {"type": ["number", "null"]},
        "arv": {"type": ["string", "number", "null"]},
        "spread": {"type": ["string", "number", "null"]},
        "pct_arv": {"type": ["string", "number", "null"]},
        "batchrank_score_category": {"type": ["string", "null"]},
        "tag_names": {"type": ["string", "null"]},
        "foreclosure_doc_type": {"type": ["string", "null"]},
        "foreclosure_status": {"type": ["string", "null"]},
        "foreclosure_auction_date": {"type": ["string", "null", "object"]},
        "foreclosure_loan_default_date": {"type": ["string", "null", "object"]},
        "foreclosure_recording_date": {"type": ["string", "null", "object"]},
        "foreclosure_case_number": {"type": ["string", "null"]},
        "foreclosure_trustee_attorney": {"type": ["string", "null"]},
        "mls_listing_date": {"type": ["string", "null", "object"]},
        "mls_listing_amount": {"type": ["number", "null"]}
    },
    "required": ["address", "city"],  # Only require basic address info
    "additionalProperties": True
}

# Update the schema registry
SCHEMA_REGISTRY = {
    "mls": "MLS_SCHEMA",
    "mapps": "MAPPS_SCHEMA", 
    "batch_leads": "BATCH_LEADS_SCHEMA",
    "ecourt": "ECOURT_SCHEMA",
    "rpt": "RPT_SCHEMA"
}

def get_schema(file_type):
    """Get the appropriate schema for a file type"""
    if file_type not in SCHEMA_REGISTRY:
        raise ValueError(f"Unknown file type: {file_type}")
    
    schema_name = SCHEMA_REGISTRY[file_type]
    return globals()[schema_name]

def validate_against_schema(data, file_type):
    """Validate data against the appropriate schema"""
    import jsonschema
    from datetime import datetime, date
    
    schema = get_schema(file_type)
    
    # Convert datetime objects to strings for validation
    def convert_datetime(obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return obj
    
    # Convert NaN values to None
    def convert_nan(obj):
        if hasattr(obj, '__iter__') and not isinstance(obj, str):
            try:
                return [None if pd.isna(x) else convert_datetime(x) for x in obj]
            except:
                return obj
        else:
            try:
                return None if pd.isna(obj) else convert_datetime(obj)
            except:
                return obj
    
    try:
        # Convert the data
        if isinstance(data, dict):
            converted_data = {k: convert_nan(v) for k, v in data.items()}
        else:
            converted_data = convert_nan(data)
            
        jsonschema.validate(converted_data, schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Validation error: {str(e)}"

