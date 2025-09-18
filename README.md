# 🏠 Augur Seller Scoring Application

A comprehensive browser application for predicting which homeowners are most likely to sell their homes in the next 6-8 months using MLS data, building permits, legal events, and other proprietary data sources.

## 🎯 Features

### ✅ Core Functionality
- **Upload & Validation**: Streamlit file uploads for MLS, MAPPS, Batch Leads, and eCourt/BOC data
- **JSON Schema Validation**: Strict validation with clear error messages
- **Idempotent Ingestion**: PostgreSQL database with upsert functionality and change reports
- **Rule-based Scoring**: Configurable weights with time decay for multiple signals
- **Optuna Optimization**: Hyperparameter tuning with time-aware cross-validation
- **Backtesting**: Precision@K, PR-AUC, and calibration analysis
- **Export Functionality**: Top-K CSV and GHL dry-run CSV exports
- **Docker Containerization**: Complete deployment with PostgreSQL

### 📊 Scoring Signals
- **Tenure**: Property ownership duration and equity
- **Legal Events**: Probate, divorce, lis pendens, foreclosure
- **Permit Activity**: Building permits and renovation activity
- **Nearby Listings**: Market activity in the area
- **Maintenance Burden**: Property characteristics affecting selling likelihood

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- 4GB+ RAM available
- Ports 8501 and 5432 available

### Installation

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd augur_app
   ```

2. **Environment Configuration**
   ```bash
   cp env.example .env
   # Edit .env with your database credentials
   ```

3. **Start with Docker Compose**
   ```bash
   docker-compose up -d
   ```

4. **Access Application**
   - Main App: http://localhost:8501
   - Database: localhost:5432
   - pgAdmin (optional): http://localhost:5050

### Manual Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup PostgreSQL Database**
   ```bash
   # Create database and run schema
   psql -U postgres -c "CREATE DATABASE augur;"
   psql -U postgres -d augur -f db/complete_schema.sql
   ```

3. **Configure Environment**
   ```bash
   cp env.example .env
   # Update database credentials in .env
   ```

4. **Run Application**
   ```bash
   streamlit run main_app.py
   ```

## 📁 Project Structure

```
augur_app/
├── main_app.py              # Main Streamlit application
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Multi-container setup
├── env.example             # Environment variables template
├── README.md               # This file
│
├── schemas/                # Data validation
│   ├── json_schemas.py     # JSON schemas for all file types
│   ├── normalizers.py      # Data normalization
│   └── validators.py       # Data validation
│
├── ingest/                 # Database ingestion
│   └── ingest.py          # Upsert functions
│
├── scoring/               # Scoring engine
│   ├── engine.py          # Main scoring logic
│   └── optuna_optimizer.py # Hyperparameter optimization
│
├── backtest/              # Backtesting
│   └── backtester.py      # Performance analysis
│
├── utils/                 # Utilities
│   └── exporter.py        # Export functionality
│
├── db/                    # Database
│   ├── complete_schema.sql # Full database schema
│   └── tables.sql         # Basic tables
│
├── uploads/               # Uploaded files
└── exports/               # Exported results
```

## 🔧 Usage Guide

### 1. Upload & Validate Data

1. Navigate to "📤 Upload & Validate"
2. Upload CSV/Excel files (MLS, MAPPS, Batch Leads, eCourt)
3. Files are automatically validated against JSON schemas
4. Review validation results and normalization messages
5. Click "Ingest to Database" to store data

**Supported File Types:**
- **MLS**: Property listings with MLS numbers, prices, status
- **MAPPS**: Building permits with case numbers, dates, addresses
- **Batch Leads**: Owner information with contact details
- **eCourt**: Legal events with case numbers, types, parties

### 2. Configure Scoring Engine

1. Navigate to "🎯 Scoring Engine"
2. Adjust weight sliders for different signals:
   - Tenure Weight (0.0-1.0)
   - Equity Weight (0.0-1.0)
   - Legal Events Weight (0.0-1.0)
   - Permit Activity Weight (0.0-1.0)
   - Nearby Listings Weight (0.0-1.0)
   - Maintenance Burden Weight (0.0-1.0)
3. Set time decay half-life (30-365 days)
4. Click "Generate Scores" to calculate seller likelihood

### 3. Optimize Weights with Optuna

1. Navigate to "🔧 Weight Tuning"
2. Configure optimization parameters:
   - Number of trials (10-500)
   - Timeout in minutes (5-120)
3. Click "Calculate Baseline Score" to see tenure-only performance
4. Click "Tune Weights" to run optimization
5. Review best configuration and optimization progress

### 4. Backtest Performance

1. Navigate to "📊 Backtesting"
2. Select backtest date range
3. Configure model weights
4. Click "Run Backtest" to analyze performance
5. Review metrics:
   - Precision@K (K=10, 25, 50, 100, 200)
   - PR-AUC and ROC-AUC
   - Calibration analysis
   - Baseline comparison

### 5. Export Results

1. Navigate to "📁 Export Results"
2. Configure export settings:
   - Number of top properties (10-1000)
   - Export format (Top-K CSV, GHL Dry-Run, Both)
3. Set model weights
4. Click "Generate Export"
5. Download CSV files for downstream use

## 📊 Data Schemas

### MLS Data Schema
```json
{
  "MLS #": "string (numeric)",
  "Status": "ACT|SLD|EXP|CXL|PND|WTH",
  "Type": "SF|Single Family|Condo|Townhouse",
  "L Price": "string (price format)",
  "Address": "string (min 5 chars)",
  "District": "string",
  "Bds": "integer",
  "Bths": "number",
  "Liv-SF": "integer",
  "City": "string",
  "State": "HI|Hawaii",
  "Zip": "string (5 digits)"
}
```

### MAPPS Permits Schema
```json
{
  "Case Number": "string (required)",
  "Type": "string",
  "Status": "Completed|In Review|Pending|Approved|Denied",
  "Address": "string (min 5 chars)",
  "Issued Date": "MM/DD/YYYY format",
  "Applied Date": "MM/DD/YYYY format",
  "Expiration Date": "MM/DD/YYYY format",
  "Finalized Date": "MM/DD/YYYY format"
}
```

### Batch Leads Schema
```json
{
  "Owner Name": "string (required)",
  "Property Address": "string (required)",
  "City": "string (required)",
  "State": "HI|Hawaii (required)",
  "ZIP": "string (5 digits, required)",
  "TMK/APN": "string",
  "Phone": "string (phone format)",
  "Email": "string (email format)",
  "Equity": "number (percentage)"
}
```

### eCourt Legal Events Schema
```json
{
  "Case Number": "string (required)",
  "Case Type": "Probate|Divorce|Lis Pendens|Foreclosure|Bankruptcy|Other",
  "Status": "Open|Closed|Pending|Dismissed",
  "Party Name": "string (required)",
  "Address": "string (required)",
  "Filing Date": "MM/DD/YYYY format",
  "TMK/APN": "string"
}
```

## 🎯 Scoring Algorithm

### Signal Components

1. **Tenure Score**: Based on property equity and ownership duration
2. **Equity Score**: Higher equity = higher selling likelihood
3. **Legal Score**: Probate (0.9), Divorce (0.8), Lis Pendens (0.7), etc.
4. **Permit Score**: Recent building activity suggests potential sale
5. **Listing Score**: Nearby market activity influences selling probability
6. **Maintenance Score**: Property characteristics affecting maintenance burden

### Time Decay Formula
```
decay_factor = 2^(-days_ago / half_life_days)
```

### Final Score Calculation
```
total_score = Σ(signal_score * weight * decay_factor)
```

## 🔧 Configuration

### Environment Variables
```bash
# Database
DB_USER=Augur_App
DB_PASSWORD=your_password
DB_HOST=localhost
DB_NAME=augur
DB_PORT=5432

# Application
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Scoring Defaults
DEFAULT_TENURE_WEIGHT=0.3
DEFAULT_EQUITY_WEIGHT=0.25
DEFAULT_LEGAL_WEIGHT=0.2
DEFAULT_PERMIT_WEIGHT=0.15
DEFAULT_LISTING_WEIGHT=0.1

# Time Decay
TIME_DECAY_HALF_LIFE_DAYS=90

# Optuna
OPTUNA_N_TRIALS=200
OPTUNA_TIMEOUT_SECONDS=3600

# Export
TOP_K_DEFAULT=100
```

## 📈 Performance Metrics

### Backtesting Metrics
- **Precision@K**: Accuracy of top K predictions
- **PR-AUC**: Precision-Recall Area Under Curve
- **ROC-AUC**: Receiver Operating Characteristic AUC
- **Calibration Score**: How well predicted probabilities match actual rates
- **Brier Score**: Mean squared error of probability predictions

### Optimization Objective
Maximize: `0.7 * Precision@K + 0.3 * PR-AUC`

## 🐳 Docker Commands

### Start Services
```bash
# Start all services
docker-compose up -d

# Start with pgAdmin
docker-compose --profile admin up -d

# View logs
docker-compose logs -f augur_app
```

### Database Management
```bash
# Connect to database
docker-compose exec postgres psql -U Augur_App -d augur

# Backup database
docker-compose exec postgres pg_dump -U Augur_App augur > backup.sql

# Restore database
docker-compose exec -T postgres psql -U Augur_App -d augur < backup.sql
```

### Application Management
```bash
# Restart application
docker-compose restart augur_app

# Update application
docker-compose build augur_app
docker-compose up -d augur_app

# View application logs
docker-compose logs -f augur_app
```

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check PostgreSQL is running: `docker-compose ps`
   - Verify credentials in `.env` file
   - Check database logs: `docker-compose logs postgres`

2. **File Upload Errors**
   - Ensure file format matches expected schema
   - Check file size limits
   - Verify required columns are present

3. **Scoring Errors**
   - Ensure data is ingested in database
   - Check for missing required fields
   - Verify database connectivity

4. **Export Issues**
   - Check exports directory permissions
   - Ensure sufficient disk space
   - Verify data validation passed

### Logs and Debugging
```bash
# Application logs
docker-compose logs augur_app

# Database logs
docker-compose logs postgres

# All services logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f
```

## 📞 Support

For technical support or questions:
- Check the troubleshooting section above
- Review application logs for error details
- Ensure all prerequisites are met
- Verify database connectivity and data integrity

## 🔄 Updates and Maintenance

### Regular Maintenance
1. **Database Backups**: Run daily backups of PostgreSQL data
2. **Log Rotation**: Monitor and rotate application logs
3. **Performance Monitoring**: Check scoring performance and optimization results
4. **Data Validation**: Regularly validate incoming data quality

### Version Updates
1. **Backup Data**: Always backup database before updates
2. **Test Environment**: Test updates in staging environment first
3. **Schema Migrations**: Run any required database migrations
4. **Configuration Updates**: Update environment variables as needed

---

**Built with ❤️ for real estate professionals**
#   p a u l  
 