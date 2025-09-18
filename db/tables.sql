-- db/tables.sql

-- 1️⃣ Properties table (MLS + Batch Leads)
CREATE TABLE IF NOT EXISTS properties (
    id SERIAL PRIMARY KEY,
    mls_id VARCHAR(50),
    owner_name VARCHAR(255),
    property_address TEXT NOT NULL,
    city VARCHAR(100),
    state CHAR(2),
    zip_code CHAR(5),
    district VARCHAR(100),
    bedrooms INT,
    bathrooms FLOAT,
    living_area INT,
    piccount INT,
    list_price NUMERIC(12,2),
    status VARCHAR(50),
    report_date DATE,
    buyer_agent VARCHAR(255),
    equity NUMERIC(5,2),
    tmk_apn VARCHAR(50),
    phone VARCHAR(50),
    email VARCHAR(100),
    UNIQUE (mls_id, tmk_apn, report_date)
);

-- 2️⃣ Permits table (MAPPS)
CREATE TABLE IF NOT EXISTS permits (
    id SERIAL PRIMARY KEY,
    case_number VARCHAR(50) NOT NULL,
    permit_type TEXT,
    status VARCHAR(50),
    project_name TEXT,
    issued_date DATE,
    applied_date DATE,
    expiration_date DATE,
    finalized_date DATE,
    module_name TEXT,
    address TEXT,
    parcel_number VARCHAR(50),
    description TEXT,
    UNIQUE(case_number)
);

-- 3️⃣ Legal events table (eCourt)
CREATE TABLE IF NOT EXISTS legal_events (
    id SERIAL PRIMARY KEY,
    case_number VARCHAR(50) NOT NULL,
    filing_date DATE,
    case_type TEXT,
    status VARCHAR(50),
    party_name TEXT,
    address TEXT,
    tmk_apn VARCHAR(50),
    UNIQUE(case_number)
);
