-- Migration script to update the properties table
-- Run this to apply the latest schema changes

-- Make property_address nullable
ALTER TABLE properties ALTER COLUMN property_address DROP NOT NULL;

-- Add file_type column if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'properties' AND column_name = 'file_type') THEN
        ALTER TABLE properties ADD COLUMN file_type VARCHAR(50);
    END IF;
END $$;

-- Add unique constraint on mls_id if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints 
                   WHERE table_name = 'properties' AND constraint_name = 'properties_mls_id_key') THEN
        ALTER TABLE properties ADD CONSTRAINT properties_mls_id_key UNIQUE (mls_id);
    END IF;
END $$;

-- Update the existing unique constraint to allow both mls_id and tmk_apn+report_date
-- First drop the existing constraint if it exists
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.table_constraints 
               WHERE table_name = 'properties' AND constraint_name = 'properties_tmk_apn_report_date_key') THEN
        ALTER TABLE properties DROP CONSTRAINT properties_tmk_apn_report_date_key;
    END IF;
END $$;

-- Add the new constraint
ALTER TABLE properties ADD CONSTRAINT properties_tmk_apn_report_date_key UNIQUE (tmk_apn, report_date);


