-- Migration to fix address column name mismatch
-- This changes property_address to address to match the code

-- Rename the column
ALTER TABLE properties RENAME COLUMN property_address TO address;

-- Update any indexes that reference the old column name
-- (PostgreSQL will automatically update the index when we rename the column)

