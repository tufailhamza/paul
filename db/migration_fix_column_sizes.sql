-- Migration to fix column sizes for Batch Leads data
-- Fix zip_code to handle extended ZIP codes (e.g., 96753-7440)
-- Fix equity to handle larger values

-- Fix zip_code column size
ALTER TABLE properties ALTER COLUMN zip_code TYPE VARCHAR(10);

-- Fix equity column size  
ALTER TABLE properties ALTER COLUMN equity TYPE NUMERIC(12,2);

-- Add comment
COMMENT ON COLUMN properties.zip_code IS 'ZIP code, can include extended format (e.g., 96753-7440)';
COMMENT ON COLUMN properties.equity IS 'Property equity value, can be negative for underwater properties';


