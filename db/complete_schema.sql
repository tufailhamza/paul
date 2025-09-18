-- Complete Database Schema for Augur Seller Scoring App
-- This includes all tables needed for the production application

-- ==============================================
-- 1. PROPERTIES TABLE (MLS + Batch Leads data)
-- ==============================================
CREATE TABLE IF NOT EXISTS properties (
    id SERIAL PRIMARY KEY,
    mls_id VARCHAR(50),
    owner_name VARCHAR(255),
    address TEXT,
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
    file_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (mls_id),
    UNIQUE (tmk_apn, report_date)
);

-- ==============================================
-- 2. PERMITS TABLE (MAPPS data)
-- ==============================================
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(case_number)
);

-- ==============================================
-- 3. LEGAL EVENTS TABLE (eCourt/BOC data)
-- ==============================================
CREATE TABLE IF NOT EXISTS legal_events (
    id SERIAL PRIMARY KEY,
    case_number VARCHAR(50) NOT NULL,
    filing_date DATE,
    case_type TEXT,
    status VARCHAR(50),
    party_name TEXT,
    address TEXT,
    tmk_apn VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(case_number)
);

-- ==============================================
-- 4. SCORING RESULTS TABLE
-- ==============================================
CREATE TABLE IF NOT EXISTS scoring_results (
    id SERIAL PRIMARY KEY,
    property_id INT REFERENCES properties(id),
    tmk_apn VARCHAR(50),
    address TEXT,
    score NUMERIC(10,4),
    tenure_score NUMERIC(10,4),
    equity_score NUMERIC(10,4),
    legal_score NUMERIC(10,4),
    permit_score NUMERIC(10,4),
    listing_score NUMERIC(10,4),
    maintenance_score NUMERIC(10,4),
    total_score NUMERIC(10,4),
    rank_position INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50),
    weights_config JSONB
);

-- ==============================================
-- 5. MODEL CONFIGURATIONS TABLE
-- ==============================================
CREATE TABLE IF NOT EXISTS model_configurations (
    id SERIAL PRIMARY KEY,
    config_name VARCHAR(100) NOT NULL,
    tenure_weight NUMERIC(5,4),
    equity_weight NUMERIC(5,4),
    legal_weight NUMERIC(5,4),
    permit_weight NUMERIC(5,4),
    listing_weight NUMERIC(5,4),
    maintenance_weight NUMERIC(5,4),
    time_decay_half_life_days INT,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    performance_metrics JSONB
);

-- ==============================================
-- 6. BACKTEST RESULTS TABLE
-- ==============================================
CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    model_config_id INT REFERENCES model_configurations(id),
    backtest_date DATE,
    precision_at_k NUMERIC(5,4),
    recall_at_k NUMERIC(5,4),
    pr_auc NUMERIC(5,4),
    calibration_score NUMERIC(5,4),
    top_k_count INT,
    total_properties INT,
    time_period_start DATE,
    time_period_end DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==============================================
-- 7. EXPORT LOGS TABLE
-- ==============================================
CREATE TABLE IF NOT EXISTS export_logs (
    id SERIAL PRIMARY KEY,
    export_type VARCHAR(50), -- 'top_k_csv', 'ghl_dry_run'
    export_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    record_count INT,
    file_path TEXT,
    model_config_id INT REFERENCES model_configurations(id),
    parameters JSONB
);

-- ==============================================
-- 8. OPTUNA TRIALS TABLE (for tracking optimization)
-- ==============================================
CREATE TABLE IF NOT EXISTS optuna_trials (
    id SERIAL PRIMARY KEY,
    trial_number INT,
    objective_value NUMERIC(10,6),
    tenure_weight NUMERIC(5,4),
    equity_weight NUMERIC(5,4),
    legal_weight NUMERIC(5,4),
    permit_weight NUMERIC(5,4),
    listing_weight NUMERIC(5,4),
    maintenance_weight NUMERIC(5,4),
    time_decay_half_life_days INT,
    trial_status VARCHAR(20), -- 'COMPLETE', 'PRUNED', 'FAIL'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==============================================
-- 9. INDEXES FOR PERFORMANCE
-- ==============================================

-- Properties table indexes
CREATE INDEX IF NOT EXISTS idx_properties_tmk_apn ON properties(tmk_apn);
CREATE INDEX IF NOT EXISTS idx_properties_address ON properties(property_address);
CREATE INDEX IF NOT EXISTS idx_properties_status ON properties(status);
CREATE INDEX IF NOT EXISTS idx_properties_report_date ON properties(report_date);
CREATE INDEX IF NOT EXISTS idx_properties_city ON properties(city);

-- Permits table indexes
CREATE INDEX IF NOT EXISTS idx_permits_address ON permits(address);
CREATE INDEX IF NOT EXISTS idx_permits_parcel_number ON permits(parcel_number);
CREATE INDEX IF NOT EXISTS idx_permits_issued_date ON permits(issued_date);
CREATE INDEX IF NOT EXISTS idx_permits_status ON permits(status);

-- Legal events table indexes
CREATE INDEX IF NOT EXISTS idx_legal_events_tmk_apn ON legal_events(tmk_apn);
CREATE INDEX IF NOT EXISTS idx_legal_events_address ON legal_events(address);
CREATE INDEX IF NOT EXISTS idx_legal_events_filing_date ON legal_events(filing_date);
CREATE INDEX IF NOT EXISTS idx_legal_events_case_type ON legal_events(case_type);

-- Scoring results indexes
CREATE INDEX IF NOT EXISTS idx_scoring_results_tmk_apn ON scoring_results(tmk_apn);
CREATE INDEX IF NOT EXISTS idx_scoring_results_score ON scoring_results(total_score DESC);
CREATE INDEX IF NOT EXISTS idx_scoring_results_rank ON scoring_results(rank_position);
CREATE INDEX IF NOT EXISTS idx_scoring_results_created_at ON scoring_results(created_at);

-- ==============================================
-- 10. TRIGGERS FOR UPDATED_AT
-- ==============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to all tables with updated_at
CREATE TRIGGER update_properties_updated_at BEFORE UPDATE ON properties
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_permits_updated_at BEFORE UPDATE ON permits
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_legal_events_updated_at BEFORE UPDATE ON legal_events
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ==============================================
-- 11. VIEWS FOR COMMON QUERIES
-- ==============================================

-- View for properties with latest scores
CREATE OR REPLACE VIEW properties_with_scores AS
SELECT 
    p.*,
    sr.total_score,
    sr.rank_position,
    sr.tenure_score,
    sr.equity_score,
    sr.legal_score,
    sr.permit_score,
    sr.listing_score,
    sr.maintenance_score
FROM properties p
LEFT JOIN scoring_results sr ON p.tmk_apn = sr.tmk_apn
WHERE sr.created_at = (
    SELECT MAX(created_at) 
    FROM scoring_results sr2 
    WHERE sr2.tmk_apn = p.tmk_apn
);

-- View for top scoring properties
CREATE OR REPLACE VIEW top_scoring_properties AS
SELECT 
    p.*,
    sr.total_score,
    sr.rank_position
FROM properties p
JOIN scoring_results sr ON p.tmk_apn = sr.tmk_apn
WHERE sr.created_at = (
    SELECT MAX(created_at) 
    FROM scoring_results sr2 
    WHERE sr2.tmk_apn = p.tmk_apn
)
ORDER BY sr.total_score DESC, sr.rank_position ASC;

-- ==============================================
-- 12. INITIAL DATA
-- ==============================================

-- Insert default model configuration
INSERT INTO model_configurations (
    config_name, 
    tenure_weight, 
    equity_weight, 
    legal_weight, 
    permit_weight, 
    listing_weight, 
    maintenance_weight,
    time_decay_half_life_days,
    is_active
) VALUES (
    'default_config',
    0.30,
    0.25,
    0.20,
    0.15,
    0.10,
    0.00,
    90,
    TRUE
) ON CONFLICT DO NOTHING;
