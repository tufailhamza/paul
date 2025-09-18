-- Migration to add Batch Leads fields to properties table
-- This adds all the fields we need from Batch Leads Excel files

-- Add new columns for Batch Leads data
ALTER TABLE properties ADD COLUMN IF NOT EXISTS first_name VARCHAR(100);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS last_name VARCHAR(100);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS mailing_address TEXT;
ALTER TABLE properties ADD COLUMN IF NOT EXISTS mailing_city VARCHAR(100);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS mailing_state CHAR(2);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS mailing_zip VARCHAR(10);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS mailing_county VARCHAR(100);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS property_zip VARCHAR(10);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS is_vacant VARCHAR(10);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS property_type_detail VARCHAR(100);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS owner_occupied VARCHAR(10);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS lot_size_sqft INT;
ALTER TABLE properties ADD COLUMN IF NOT EXISTS year_built INT;
ALTER TABLE properties ADD COLUMN IF NOT EXISTS total_assessed_value NUMERIC(12,2);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS zoning_code VARCHAR(20);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS last_sale_date DATE;
ALTER TABLE properties ADD COLUMN IF NOT EXISTS last_sale_price NUMERIC(12,2);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS total_loan_balance NUMERIC(12,2);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS estimated_value NUMERIC(12,2);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS ltv_current_combined NUMERIC(5,2);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS self_managed VARCHAR(10);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS loan_recording_date DATE;
ALTER TABLE properties ADD COLUMN IF NOT EXISTS loan_type VARCHAR(100);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS loan_amount NUMERIC(12,2);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS loan_lender_name VARCHAR(255);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS loan_due_date DATE;
ALTER TABLE properties ADD COLUMN IF NOT EXISTS loan_est_payment NUMERIC(10,2);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS loan_est_interest_rate NUMERIC(5,2);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS loan_est_balance NUMERIC(12,2);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS loan_term_months INT;
ALTER TABLE properties ADD COLUMN IF NOT EXISTS arv NUMERIC(12,2);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS spread NUMERIC(12,2);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS pct_arv NUMERIC(5,2);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS batchrank_score_category VARCHAR(50);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS tag_names TEXT;
ALTER TABLE properties ADD COLUMN IF NOT EXISTS foreclosure_doc_type VARCHAR(100);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS foreclosure_status VARCHAR(100);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS foreclosure_auction_date DATE;
ALTER TABLE properties ADD COLUMN IF NOT EXISTS foreclosure_loan_default_date DATE;
ALTER TABLE properties ADD COLUMN IF NOT EXISTS foreclosure_recording_date DATE;
ALTER TABLE properties ADD COLUMN IF NOT EXISTS foreclosure_case_number VARCHAR(100);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS foreclosure_trustee_attorney VARCHAR(255);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS mls_listing_date DATE;
ALTER TABLE properties ADD COLUMN IF NOT EXISTS mls_listing_amount NUMERIC(12,2);

-- Update the equity column to handle larger values from Batch Leads
ALTER TABLE properties ALTER COLUMN equity TYPE NUMERIC(12,2);

-- Update the tmk_apn constraint to handle NULL values properly
ALTER TABLE properties DROP CONSTRAINT IF EXISTS properties_tmk_apn_report_date_key;
ALTER TABLE properties ADD CONSTRAINT properties_tmk_apn_report_date_unique 
    UNIQUE (tmk_apn, report_date) WHERE (tmk_apn IS NOT NULL);

-- Add indexes for new fields
CREATE INDEX IF NOT EXISTS idx_properties_first_name ON properties(first_name);
CREATE INDEX IF NOT EXISTS idx_properties_last_name ON properties(last_name);
CREATE INDEX IF NOT EXISTS idx_properties_year_built ON properties(year_built);
CREATE INDEX IF NOT EXISTS idx_properties_equity ON properties(equity);
CREATE INDEX IF NOT EXISTS idx_properties_owner_occupied ON properties(owner_occupied);
CREATE INDEX IF NOT EXISTS idx_properties_last_sale_date ON properties(last_sale_date);
CREATE INDEX IF NOT EXISTS idx_properties_ltv ON properties(ltv_current_combined);
CREATE INDEX IF NOT EXISTS idx_properties_batchrank ON properties(batchrank_score_category);

-- Update the address index to use the correct column name
DROP INDEX IF EXISTS idx_properties_address;
CREATE INDEX IF NOT EXISTS idx_properties_address ON properties(address);

