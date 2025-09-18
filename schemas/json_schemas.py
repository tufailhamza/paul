"""
JSON Schemas for data validation
These schemas enforce strict validation as required by the client
"""

import json
from typing import Dict, Any

# ==============================================
# MLS DATA SCHEMA - Unified schema for all MLS file types
# ==============================================
MLS_SCHEMA = {
    "type": "object",
    "properties": {
        "mls_id": {
            "type": ["string", "integer", "null"],
            "description": "MLS number"
        },
        "piccount": {
            "type": ["integer", "string"],
            "description": "Number of pictures"
        },
        "status": {
            "type": ["string", "null", "object"],
            "enum": ["ACT", "SLD", "EXP", "CXL", "PND", "WTH", "UNKNOWN", None],
            "description": "Property status"
        },
        "property_type": {
            "type": "string",
            "enum": ["SF", "Single Family", "Condo", "Townhouse"],
            "description": "Property type"
        },
        "list_price": {
            "type": "string",
            "description": "List price with dollar sign and commas"
        },
        "address": {
            "type": "string",
            "minLength": 5,
            "description": "Property address"
        },
        "district": {
            "type": "string",
            "description": "Property district"
        },
        "bedrooms": {
            "type": ["integer", "string"],
            "description": "Number of bedrooms"
        },
        "bathrooms": {
            "type": ["number", "string"],
            "description": "Number of bathrooms"
        },
        "living_area": {
            "type": ["integer", "string"],
            "description": "Living area in square feet (may contain commas)"
        },
        "city": {
            "type": "string",
            "description": "City name"
        },
        "state": {
            "type": "string",
            "enum": ["HI", "Hawaii"],
            "description": "State abbreviation or name"
        },
        "zip_code": {
            "type": "string",
            "pattern": "^[0-9]{5}$",
            "description": "5-digit ZIP code"
        },
        "Report Date": {
            "type": "string",
            "description": "Report date"
        },
        "Agent - Agt Nm Ph": {
            "type": ["string", "null", "object"],
            "description": "Agent name and phone"
        },
        "Buys Agt - Agt Name": {
            "type": ["string", "null", "object"],
            "description": "Buyer agent name"
        },
        "City ": {
            "type": "string",
            "description": "City name (with trailing space)"
        },
        "View": {
            "type": ["string", "null", "object"],
            "description": "Property view"
        },
        "Wtrfrt": {
            "type": ["string", "null", "object"],
            "description": "Waterfront indicator"
        },
        "mls_id": {
            "type": ["string", "integer", "null"],
            "description": "MLS number (normalized)"
        },
        "piccount": {
            "type": ["integer", "string"],
            "description": "Number of pictures (normalized)"
        },
        "status": {
            "type": ["string", "null", "object"],
            "enum": ["ACT", "SLD", "EXP", "CXL", "PND", "WTH", "UNKNOWN", None],
            "description": "Property status (normalized)"
        },
        "property_type": {
            "type": "string",
            "enum": ["SF", "Single Family", "Condo", "Townhouse"],
            "description": "Property type (normalized)"
        },
        "list_price": {
            "type": ["number", "string"],
            "description": "List price (normalized)"
        },
        "address": {
            "type": "string",
            "minLength": 5,
            "description": "Property address (normalized)"
        },
        "district": {
            "type": "string",
            "description": "Property district (normalized)"
        },
        "bedrooms": {
            "type": ["integer", "string"],
            "description": "Number of bedrooms (normalized)"
        },
        "bathrooms": {
            "type": ["number", "string"],
            "description": "Number of bathrooms (normalized)"
        },
        "living_area": {
            "type": ["integer", "string"],
            "description": "Living area in square feet (normalized)"
        },
        "city": {
            "type": "string",
            "description": "City name (normalized)"
        },
        "state": {
            "type": "string",
            "description": "State abbreviation (normalized)"
        },
        "zip_code": {
            "type": "string",
            "pattern": "^[0-9]{5}$",
            "description": "5-digit ZIP code (normalized)"
        },
        "report_date": {
            "type": ["string", "null", "object"],
            "description": "Report date (normalized)"
        },
        "agent_name": {
            "type": ["string", "null", "object"],
            "description": "Agent name and phone (normalized)"
        },
        "buyer_agent": {
            "type": ["string", "null", "object"],
            "description": "Buyer agent name (normalized)"
        },
        "view": {
            "type": ["string", "null", "object"],
            "description": "Property view (normalized)"
        },
        "wtrfrt": {
            "type": ["string", "null", "object"],
            "description": "Waterfront indicator (normalized)"
        },
        "tmk_apn": {
            "type": ["string", "null", "object"],
            "description": "Tax map key or assessor parcel number (normalized)"
        }
    },
    "required": ["status", "property_type", "list_price", "address", "district", "bedrooms", "bathrooms", "living_area", "city", "state", "zip_code"],
    "additionalProperties": True
}

# ==============================================
# MAPPS PERMITS SCHEMA
# ==============================================
MAPPS_SCHEMA = {
    "type": "object",
    "properties": {
        "case_number": {
            "type": "string",
            "minLength": 1,
            "description": "Unique case number"
        },
        "permit_type": {
            "type": "string",
            "description": "Permit type"
        },
        "status": {
            "type": "string",
            "enum": ["Completed", "In Review", "Pending", "Approved", "Denied"],
            "description": "Permit status"
        },
        "project_name": {
            "type": ["string", "null", "object"],
            "description": "Project name"
        },
        "issued_date": {
            "type": ["string", "null", "object"],
            "description": "Date permit was issued"
        },
        "applied_date": {
            "type": ["string", "null", "object"],
            "description": "Date permit was applied for"
        },
        "expiration_date": {
            "type": ["string", "null", "object"],
            "description": "Date permit expires"
        },
        "finalized_date": {
            "type": ["string", "null", "object"],
            "description": "Date permit was finalized"
        },
        "module_name": {
            "type": ["string", "null", "object"],
            "description": "Module name"
        },
        "address": {
            "type": "string",
            "minLength": 5,
            "description": "Property address"
        },
        "parcel_number": {
            "type": ["string", "null", "object"],
            "description": "Main parcel number"
        },
        "description": {
            "type": ["string", "null", "object"],
            "description": "Project description"
        }
    },
    "required": ["case_number", "permit_type", "status", "address"],
    "additionalProperties": False
}

# ==============================================
# BATCH LEADS SCHEMA
# ==============================================
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

# ==============================================
# ECOURT/LEGAL EVENTS SCHEMA
# ==============================================
ECOURT_SCHEMA = {
    "type": "object",
    "properties": {
        "Case Number": {
            "type": "string",
            "minLength": 1,
            "description": "Unique case number"
        },
        "Filing Date": {
            "type": ["string", "null", "object"],
            "pattern": "^(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/([0-9]{4})$",
            "description": "Filing date in MM/DD/YYYY format"
        },
        "Case Type": {
            "type": "string",
            "enum": ["Probate", "Divorce", "Lis Pendens", "Foreclosure", "Bankruptcy", "Other"],
            "description": "Type of legal case"
        },
        "Status": {
            "type": "string",
            "enum": ["Open", "Closed", "Pending", "Dismissed"],
            "description": "Case status"
        },
        "Party Name": {
            "type": "string",
            "minLength": 1,
            "description": "Party name"
        },
        "Address": {
            "type": "string",
            "minLength": 5,
            "description": "Property address"
        },
        "TMK/APN": {
            "type": ["string", "null", "object"],
            "description": "Tax map key or assessor parcel number"
        }
    },
    "required": ["Case Number", "Case Type", "Status", "Party Name", "Address"],
    "additionalProperties": False
}

# ==============================================
# RPT (Property Records) SCHEMA
# ==============================================
RPT_SCHEMA = {
    "type": "object",
    "properties": {
        "Property Address": {
            "type": "string",
            "minLength": 5,
            "description": "Property address"
        },
        "Owner Name": {
            "type": "string",
            "description": "Property owner name"
        },
        "TMK/APN": {
            "type": "string",
            "description": "Tax map key or assessor parcel number"
        },
        "City": {
            "type": "string",
            "description": "City name"
        },
        "State": {
            "type": "string",
            "enum": ["HI", "Hawaii"],
            "description": "State"
        },
        "ZIP": {
            "type": "string",
            "pattern": "^[0-9]{5}$",
            "description": "5-digit ZIP code"
        },
        "Property Type": {
            "type": "string",
            "description": "Type of property"
        },
        "Bedrooms": {
            "type": ["integer", "string"],
            "description": "Number of bedrooms"
        },
        "Bathrooms": {
            "type": ["number", "string"],
            "description": "Number of bathrooms"
        },
        "Living Area": {
            "type": ["integer", "string"],
            "description": "Living area in square feet"
        },
        "Year Built": {
            "type": ["integer", "string"],
            "description": "Year property was built"
        },
        "Assessed Value": {
            "type": ["number", "string"],
            "description": "Assessed property value"
        }
    },
    "required": ["Property Address", "Owner Name", "City", "State", "ZIP"],
    "additionalProperties": False
}

# ==============================================
# SCHEMA REGISTRY
# ==============================================
SCHEMA_REGISTRY = {
    "mls": MLS_SCHEMA,
    "mapps": MAPPS_SCHEMA,
    "batch_leads": BATCH_LEADS_SCHEMA,
    "ecourt": ECOURT_SCHEMA,
    "rpt": RPT_SCHEMA
}

def get_schema(file_type: str) -> Dict[str, Any]:
    """Get JSON schema for a specific file type"""
    return SCHEMA_REGISTRY.get(file_type.lower(), {})

def validate_against_schema(data: Dict[str, Any], file_type: str) -> tuple[bool, list]:
    """
    Validate data against the appropriate JSON schema
    Returns (is_valid, error_messages)
    """
    import jsonschema
    from jsonschema import ValidationError
    import pandas as pd
    import numpy as np
    from datetime import date, datetime
    
    schema = get_schema(file_type)
    if not schema:
        return False, [f"No schema found for file type: {file_type}"]
    
    # Clean the data before validation - convert NaN values to None and date objects to strings
    cleaned_data = {}
    for key, value in data.items():
        if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
            cleaned_data[key] = None
        elif isinstance(value, (date, datetime)):
            # Convert date/datetime objects to ISO format strings
            cleaned_data[key] = value.isoformat()
        else:
            cleaned_data[key] = value
    
    try:
        jsonschema.validate(cleaned_data, schema)
        return True, []
    except ValidationError as e:
        return False, [f"Validation error: {e.message} at path: {'.'.join(str(p) for p in e.absolute_path)}"]
    except Exception as e:
        return False, [f"Schema validation failed: {str(e)}"]

def get_required_columns(file_type: str) -> list:
    """Get list of required columns for a file type"""
    schema = get_schema(file_type)
    return schema.get("required", [])

def get_optional_columns(file_type: str) -> list:
    """Get list of optional columns for a file type"""
    schema = get_schema(file_type)
    required = set(schema.get("required", []))
    all_props = set(schema.get("properties", {}).keys())
    return list(all_props - required)
