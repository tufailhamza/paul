"""
Main Augur Seller Scoring Application
Complete Streamlit app with all required features
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import our modules
from schemas.normalizers import validate_and_normalize
from schemas.json_schemas import validate_against_schema, get_required_columns
from ingest.ingest import upsert_properties, upsert_permits, upsert_legal_events
from scoring.engine import ScoringEngine
from scoring.optuna_optimizer import OptunaOptimizer
from backtest.backtester import Backtester
from utils.exporter import Exporter

# ==============================================
# PAGE CONFIGURATION
# ==============================================
st.set_page_config(
    page_title="Augur Seller Scoring",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================
# SIDEBAR CONFIGURATION
# ==============================================
st.sidebar.title("🏠 Augur Seller Scoring")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Navigate",
    ["📤 Upload & Validate", "🎯 Scoring Engine", "🔧 Weight Tuning", "📊 Backtesting", "📁 Export Results"]
)

# ==============================================
# HELPER FUNCTIONS
# ==============================================
@st.cache_data
def load_data_summary():
    """Load data summary from database"""
    try:
        from ingest.ingest import ENGINE
        with ENGINE.connect() as conn:
            # Get counts
            properties_count = pd.read_sql("SELECT COUNT(*) as count FROM properties", conn).iloc[0]['count']
            permits_count = pd.read_sql("SELECT COUNT(*) as count FROM permits", conn).iloc[0]['count']
            legal_count = pd.read_sql("SELECT COUNT(*) as count FROM legal_events", conn).iloc[0]['count']
            
            return {
                'properties': properties_count,
                'permits': permits_count,
                'legal_events': legal_count
            }
    except Exception as e:
        st.error(f"Error loading data summary: {e}")
        return {'properties': 0, 'permits': 0, 'legal_events': 0}

def initialize_engines():
    """Initialize all engines"""
    try:
        from ingest.ingest import ENGINE
        scoring_engine = ScoringEngine(ENGINE)
        optuna_optimizer = OptunaOptimizer(ENGINE)
        backtester = Backtester(ENGINE)
        exporter = Exporter(ENGINE)
        return scoring_engine, optuna_optimizer, backtester, exporter
    except Exception as e:
        st.error(f"Error initializing engines: {e}")
        return None, None, None, None

# ==============================================
# UPLOAD & VALIDATION PAGE
# ==============================================
def upload_validation_page():
    st.title("📤 Upload & Validate Data")
    st.markdown("Upload MLS, MAPPS, Batch Leads, or eCourt/BOC files for validation and ingestion.")
    
    # Data summary
    data_summary = load_data_summary()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Properties", data_summary['properties'])
    with col2:
        st.metric("Permits", data_summary['permits'])
    with col3:
        st.metric("Legal Events", data_summary['legal_events'])
    
    st.markdown("---")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload data files",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        help="Supported formats: CSV, Excel. Files will be automatically validated against JSON schemas."
    )
    
    if uploaded_files:
        st.markdown("### 📋 File Processing Results")
        
        for file in uploaded_files:
            with st.expander(f"📄 {file.name}", expanded=True):
                # Save uploaded file
                upload_folder = "uploads"
                os.makedirs(upload_folder, exist_ok=True)
                file_path = os.path.join(upload_folder, file.name)
                
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                st.success(f"✅ File uploaded: {file.name}")
                
                # Determine file type
                fname = file.name.lower()
                if "mls" in fname:
                    file_type = "mls"
                elif "mapps" in fname or "permit" in fname:
                    file_type = "mapps"
                elif "batch" in fname or "lead" in fname:
                    file_type = "batch_leads"
                elif "court" in fname or "ecourt" in fname or "boc" in fname:
                    file_type = "ecourt"
                elif "rpt" in fname or "property" in fname:
                    file_type = "rpt"
                else:
                    file_type = "mls"  # fallback
                
                st.info(f"🔍 Detected file type: {file_type.upper()}")
                
                # Load and validate data
                try:
                    # Load data
                    if file.name.lower().endswith('.csv'):
                        df = pd.read_csv(file_path, dtype=str)
                    else:
                        df = pd.read_excel(file_path, dtype=str)
                    
                    st.write(f"📊 Loaded {len(df)} rows, {len(df.columns)} columns")
                    
                    # Normalize data first
                    st.markdown("#### 🔄 Data Normalization")
                    try:
                        df_norm, messages = validate_and_normalize(file_path, file_type)
                        
                        if messages:
                            st.warning("⚠️ Normalization messages:")
                            for msg in messages:
                                st.warning(f"  • {msg}")
                        else:
                            st.success("✅ Data normalized successfully")
                        
                        # JSON Schema validation on normalized data
                        st.markdown("#### 🔍 JSON Schema Validation")
                        validation_errors = []
                        sample_row = df_norm.iloc[0].to_dict() if len(df_norm) > 0 else {}
                        is_valid, errors = validate_against_schema(sample_row, file_type)
                        
                        if not is_valid:
                            st.error("❌ JSON Schema validation failed:")
                            for error in errors:
                                st.error(f"  • {error}")
                            validation_errors.extend(errors)
                        else:
                            st.success("✅ JSON Schema validation passed")
                        
                        # Show sample data
                        st.markdown("#### 📋 Sample Data")
                        st.dataframe(df_norm.head(10))
                        
                        # Ingest to database
                        if st.button(f"💾 Ingest {file.name} to Database", key=f"ingest_{file.name}"):
                            with st.spinner("Ingesting data..."):
                                try:
                                    if file_type in ["mls", "batch_leads", "rpt"]:
                                        result = upsert_properties(df_norm, file_type)
                                    elif file_type == "mapps":
                                        result = upsert_permits(df_norm)
                                    else:  # ecourt
                                        result = upsert_legal_events(df_norm)
                                    
                                    # Display confirmation report
                                    st.success("✅ Ingestion complete!")
                                    st.markdown("#### 📊 Confirmation Report")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Records Added", result.get('added', 0))
                                    with col2:
                                        st.metric("Records Updated", result.get('updated', 0))
                                    with col3:
                                        st.metric("Records Skipped", result.get('skipped', 0))
                                    
                                    # Display change report
                                    if 'changes' in result and result['changes']:
                                        st.markdown("#### 📋 Change Report")
                                        for change in result['changes'][:10]:  # Show first 10 changes
                                            st.write(f"**{change['address']}** ({change['tmk_apn']}):")
                                            for change_detail in change['changes']:
                                                st.write(f"  • {change_detail}")
                                        
                                        if len(result['changes']) > 10:
                                            st.write(f"... and {len(result['changes']) - 10} more changes")
                                    
                                    # Refresh data summary
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"❌ Ingestion failed: {e}")
                        
                    except Exception as e:
                        st.error(f"❌ Normalization failed: {e}")
                
                except Exception as e:
                    st.error(f"❌ File processing failed: {e}")

# ==============================================
# SCORING ENGINE PAGE
# ==============================================
def scoring_engine_page():
    st.title("🎯 Scoring Engine")
    st.markdown("Configure scoring weights and generate seller likelihood scores.")
    
    # Initialize engines
    scoring_engine, _, _, _ = initialize_engines()
    if scoring_engine is None:
        st.error("Failed to initialize scoring engine")
        return
    
    # Weight configuration
    st.markdown("### ⚖️ Scoring Weights Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Weight Sliders")
        tenure_weight = st.slider("Tenure Weight", 0.0, 1.0, 0.30, 0.05)
        equity_weight = st.slider("Equity Weight", 0.0, 1.0, 0.25, 0.05)
        legal_weight = st.slider("Legal Events Weight", 0.0, 1.0, 0.20, 0.05)
    
    with col2:
        permit_weight = st.slider("Permit Activity Weight", 0.0, 1.0, 0.15, 0.05)
        listing_weight = st.slider("Nearby Listings Weight", 0.0, 1.0, 0.10, 0.05)
        maintenance_weight = st.slider("Maintenance Burden Weight", 0.0, 1.0, 0.00, 0.05)
    
    # Time decay configuration
    st.markdown("### ⏰ Time Decay Configuration")
    time_decay_half_life = st.slider("Time Decay Half-Life (days)", 30, 365, 90, 15)
    
    # Normalize weights
    total_weight = tenure_weight + equity_weight + legal_weight + permit_weight + listing_weight + maintenance_weight
    if total_weight > 0:
        weights = {
            'tenure': tenure_weight / total_weight,
            'equity': equity_weight / total_weight,
            'legal': legal_weight / total_weight,
            'permit': permit_weight / total_weight,
            'listing': listing_weight / total_weight,
            'maintenance': maintenance_weight / total_weight
        }
    else:
        weights = {
            'tenure': 0.3, 'equity': 0.25, 'legal': 0.2,
            'permit': 0.15, 'listing': 0.1, 'maintenance': 0.0
        }
    
    # Display normalized weights
    st.markdown("#### Normalized Weights")
    weight_df = pd.DataFrame(list(weights.items()), columns=['Signal', 'Weight'])
    weight_df['Weight'] = weight_df['Weight'].round(4)
    st.dataframe(weight_df, use_container_width=True)
    
    # Generate scores
    if st.button("🎯 Generate Scores", type="primary"):
        with st.spinner("Calculating scores..."):
            try:
                # Set weights and time decay
                scoring_engine.set_weights(weights)
                scoring_engine.set_time_decay(time_decay_half_life)
                
                # Get properties to score
                from ingest.ingest import ENGINE
                with ENGINE.connect() as conn:
                    properties_df = pd.read_sql("SELECT * FROM properties WHERE status IN ('Off Market', 'Active', 'Pending', 'ACT', 'PND') OR status IS NULL", conn)
                
                if properties_df.empty:
                    st.warning("No active properties found to score")
                    return
                
                # Calculate scores
                print(f"DEBUG: UI calling scoring engine with {len(properties_df)} properties")
                print(f"DEBUG: Sample property data: {properties_df[['mls_id', 'equity', 'last_sale_date']].head().to_dict('records')}")
                scores_df = scoring_engine.calculate_total_scores(properties_df)
                
                # Save to database
                scoring_engine.save_scores_to_db(scores_df, "manual_config")
                
                st.success(f"✅ Scores calculated for {len(scores_df)} properties")
                
                # Display top results
                st.markdown("### 🏆 Top Scoring Properties")
                top_20 = scores_df.nlargest(20, 'total_score')
                display_columns = ['rank_position', 'total_score', 'property_address', 'city', 'equity', 'tenure_score', 'equity_score', 'legal_score']
                available_columns = [col for col in display_columns if col in top_20.columns]
                st.dataframe(top_20[available_columns], use_container_width=True)
                
                # Score distribution
                st.markdown("### 📊 Score Distribution")
                fig = px.histogram(scores_df, x='total_score', nbins=50, title="Score Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Score generation failed: {e}")

# ==============================================
# WEIGHT TUNING PAGE
# ==============================================
def weight_tuning_page():
    st.title("🔧 Weight Tuning with Optuna")
    st.markdown("Use Optuna hyperparameter optimization to find the best scoring weights.")
    
    # Initialize engines
    _, optuna_optimizer, _, _ = initialize_engines()
    if optuna_optimizer is None:
        st.error("Failed to initialize Optuna optimizer")
        return
    
    # Configuration
    st.markdown("### ⚙️ Optimization Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        n_trials = st.number_input("Number of Trials", min_value=10, max_value=500, value=200, step=10)
    with col2:
        timeout_minutes = st.number_input("Timeout (minutes)", min_value=5, max_value=120, value=60, step=5)
    
    # Baseline score
    st.markdown("### 📊 Baseline Performance")
    if st.button("📈 Calculate Baseline Score"):
        with st.spinner("Calculating baseline score..."):
            try:
                baseline_score = optuna_optimizer.get_baseline_score()
                st.metric("Baseline Score (Tenure Only)", f"{baseline_score:.4f}")
            except Exception as e:
                st.error(f"❌ Baseline calculation failed: {e}")
    
    # Run optimization
    st.markdown("### 🚀 Run Optimization")
    if st.button("🔧 Tune Weights", type="primary"):
        with st.spinner(f"Running {n_trials} optimization trials..."):
            try:
                # Update optimizer settings
                optuna_optimizer.n_trials = n_trials
                optuna_optimizer.timeout_seconds = timeout_minutes * 60
                
                # Run optimization
                results = optuna_optimizer.optimize()
                
                st.success("✅ Optimization completed!")
                
                # Display results
                st.markdown("### 🏆 Best Configuration Found")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best Score", f"{results['best_score']:.4f}")
                with col2:
                    st.metric("Trials Completed", results['n_trials'])
                with col3:
                    st.metric("Time Decay (days)", results['best_time_decay'])
                
                # Display best weights
                st.markdown("#### Best Weights")
                best_weights_df = pd.DataFrame(list(results['best_weights'].items()), columns=['Signal', 'Weight'])
                best_weights_df['Weight'] = best_weights_df['Weight'].round(4)
                st.dataframe(best_weights_df, use_container_width=True)
                
                # Optimization history
                st.markdown("#### Optimization History")
                study = results['study']
                trials_df = pd.DataFrame([
                    {
                        'Trial': trial.number,
                        'Score': trial.value,
                        'Status': trial.state.name
                    }
                    for trial in study.trials
                ])
                st.dataframe(trials_df, use_container_width=True)
                
                # Plot optimization progress
                if len(study.trials) > 1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(len(study.trials))),
                        y=[trial.value for trial in study.trials if trial.value is not None],
                        mode='lines+markers',
                        name='Optimization Progress'
                    ))
                    fig.update_layout(
                        title="Optimization Progress",
                        xaxis_title="Trial",
                        yaxis_title="Score"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Optimization failed: {e}")

# ==============================================
# BACKTESTING PAGE
# ==============================================
def backtesting_page():
    st.title("📊 Backtesting & Performance Analysis")
    st.markdown("Analyze model performance with precision@K, PR-AUC, and calibration metrics.")
    
    # Initialize engines
    _, _, backtester, _ = initialize_engines()
    if backtester is None:
        st.error("Failed to initialize backtester")
        return
    
    # Date range selection
    st.markdown("### 📅 Backtest Period")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    # Model configuration
    st.markdown("### ⚙️ Model Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        tenure_weight = st.slider("Tenure Weight", 0.0, 1.0, 0.30, 0.05, key="bt_tenure")
        equity_weight = st.slider("Equity Weight", 0.0, 1.0, 0.25, 0.05, key="bt_equity")
        legal_weight = st.slider("Legal Weight", 0.0, 1.0, 0.20, 0.05, key="bt_legal")
    
    with col2:
        permit_weight = st.slider("Permit Weight", 0.0, 1.0, 0.15, 0.05, key="bt_permit")
        listing_weight = st.slider("Listing Weight", 0.0, 1.0, 0.10, 0.05, key="bt_listing")
        time_decay = st.slider("Time Decay (days)", 30, 365, 90, 15, key="bt_decay")
    
    # Normalize weights
    total_weight = tenure_weight + equity_weight + legal_weight + permit_weight + listing_weight
    if total_weight > 0:
        weights = {
            'tenure': tenure_weight / total_weight,
            'equity': equity_weight / total_weight,
            'legal': legal_weight / total_weight,
            'permit': permit_weight / total_weight,
            'listing': listing_weight / total_weight,
            'maintenance': 0.0
        }
    else:
        weights = {
            'tenure': 0.3, 'equity': 0.25, 'legal': 0.2,
            'permit': 0.15, 'listing': 0.1, 'maintenance': 0.0
        }
    
    # Run backtest
    if st.button("📊 Run Backtest", type="primary"):
        with st.spinner("Running backtest analysis..."):
            try:
                # Run backtest
                results = backtester.run_backtest(
                    start_date=datetime.combine(start_date, datetime.min.time()),
                    end_date=datetime.combine(end_date, datetime.min.time()),
                    weights=weights,
                    time_decay_half_life=time_decay
                )
                
                if 'error' in results:
                    st.error(f"❌ Backtest failed: {results['error']}")
                    return
                
                st.success("✅ Backtest completed!")
                
                # Display metrics
                st.markdown("### 📈 Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Precision@100", f"{results['precision_at_k'].get(100, 0):.3f}")
                with col2:
                    st.metric("PR-AUC", f"{results['pr_auc']:.3f}")
                with col3:
                    st.metric("ROC-AUC", f"{results['roc_auc']:.3f}")
                with col4:
                    st.metric("Calibration Score", f"{1 - results['calibration_metrics']['reliability']:.3f}")
                
                # Baseline comparison
                st.markdown("### 📊 Baseline Comparison")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Model Precision@100", f"{results['precision_at_k'].get(100, 0):.3f}")
                with col2:
                    st.metric("Baseline Precision@100", f"{results['baseline_precision_at_k'].get(100, 0):.3f}")
                
                # Precision@K plot
                st.markdown("### 📊 Precision@K Analysis")
                fig = backtester.create_precision_at_k_plot(results)
                st.plotly_chart(fig, use_container_width=True)
                
                # Precision-Recall curve
                st.markdown("### 📈 Precision-Recall Curve")
                fig = backtester.create_precision_recall_plot(results)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calibration plot
                st.markdown("### 🎯 Calibration Analysis")
                fig = backtester.create_calibration_plot(results)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed metrics
                st.markdown("### 📋 Detailed Metrics")
                
                # Precision@K table
                precision_df = pd.DataFrame([
                    {'K': k, 'Precision': v, 'Baseline': results['baseline_precision_at_k'].get(k, 0)}
                    for k, v in results['precision_at_k'].items()
                ])
                st.dataframe(precision_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Backtest failed: {e}")

# ==============================================
# EXPORT RESULTS PAGE
# ==============================================
def export_results_page():
    st.title("📁 Export Results")
    st.markdown("Export top scoring properties in various formats.")
    
    # Initialize engines
    scoring_engine, _, _, exporter = initialize_engines()
    if scoring_engine is None or exporter is None:
        st.error("Failed to initialize engines")
        return
    
    # Export configuration
    st.markdown("### ⚙️ Export Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        k = st.number_input("Number of Top Properties", min_value=10, max_value=10000, value=100, step=10)
    with col2:
        export_format = st.selectbox("Export Format", ["Top-K CSV", "GHL Dry-Run CSV", "Both"])
    
    # Model configuration
    st.markdown("### ⚖️ Model Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        tenure_weight = st.slider("Tenure Weight", 0.0, 1.0, 0.30, 0.05, key="exp_tenure")
        equity_weight = st.slider("Equity Weight", 0.0, 1.0, 0.25, 0.05, key="exp_equity")
        legal_weight = st.slider("Legal Weight", 0.0, 1.0, 0.20, 0.05, key="exp_legal")
    
    with col2:
        permit_weight = st.slider("Permit Weight", 0.0, 1.0, 0.15, 0.05, key="exp_permit")
        listing_weight = st.slider("Listing Weight", 0.0, 1.0, 0.10, 0.05, key="exp_listing")
        time_decay = st.slider("Time Decay (days)", 30, 365, 90, 15, key="exp_decay")
    
    # Normalize weights
    total_weight = tenure_weight + equity_weight + legal_weight + permit_weight + listing_weight
    if total_weight > 0:
        weights = {
            'tenure': tenure_weight / total_weight,
            'equity': equity_weight / total_weight,
            'legal': legal_weight / total_weight,
            'permit': permit_weight / total_weight,
            'listing': listing_weight / total_weight,
            'maintenance': 0.0
        }
    else:
        weights = {
            'tenure': 0.3, 'equity': 0.25, 'legal': 0.2,
            'permit': 0.15, 'listing': 0.1, 'maintenance': 0.0
        }
    
    # Generate and export
    if st.button("📁 Generate Export", type="primary"):
        with st.spinner("Generating export..."):
            try:
                # Get top K properties
                top_k_df = exporter.get_top_k_properties(k=k, weights=weights, time_decay_half_life=time_decay)
                
                if top_k_df.empty:
                    st.warning("No properties found to export")
                    return
                
                st.success(f"✅ Found {len(top_k_df)} properties for export")
                
                # Validate data
                validation = exporter.validate_export_data(top_k_df)
                if not validation['is_valid']:
                    st.error("❌ Export validation failed:")
                    for error in validation['errors']:
                        st.error(f"  • {error}")
                    return
                
                if validation['warnings']:
                    st.warning("⚠️ Export warnings:")
                    for warning in validation['warnings']:
                        st.warning(f"  • {warning}")
                
                # Export files
                export_files = []
                
                if export_format in ["Top-K CSV", "Both"]:
                    csv_path = exporter.export_top_k_csv(top_k_df, k=k)
                    export_files.append(("Top-K CSV", csv_path))
                
                if export_format in ["GHL Dry-Run CSV", "Both"]:
                    ghl_path = exporter.export_ghl_dry_run_csv(top_k_df, k=k)
                    export_files.append(("GHL Dry-Run CSV", ghl_path))
                
                # Display results
                st.markdown("### 📁 Export Results")
                
                for format_name, file_path in export_files:
                    st.success(f"✅ {format_name} exported: `{file_path}`")
                    
                    # Download button - use session state to prevent re-execution
                    download_key = f"download_{format_name}_{os.path.basename(file_path)}"
                    if download_key not in st.session_state:
                        with open(file_path, 'rb') as f:
                            st.session_state[download_key] = f.read()
                    
                    st.download_button(
                        label=f"📥 Download {format_name}",
                        data=st.session_state[download_key],
                        file_name=os.path.basename(file_path),
                        mime="text/csv",
                        key=f"btn_{download_key}"
                    )
                
                # Export summary
                summary = exporter.create_export_summary(top_k_df)
                st.markdown("### 📊 Export Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", summary['total_records'])
                with col2:
                    st.metric("Score Range", f"{summary['score_range']['min']:.3f} - {summary['score_range']['max']:.3f}")
                with col3:
                    st.metric("Mean Score", f"{summary['score_range']['mean']:.3f}")
                
                # Contact information summary
                if 'contact_stats' in summary:
                    st.markdown("#### 📞 Contact Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Phone Numbers", f"{summary['contact_stats']['phone_count']} ({summary['contact_stats']['phone_percentage']:.1f}%)")
                    with col2:
                        st.metric("Email Addresses", f"{summary['contact_stats']['email_count']} ({summary['contact_stats']['email_percentage']:.1f}%)")
                
                # Show top 10 properties
                st.markdown("### 🏆 Top 10 Properties Preview")
                top_10 = top_k_df.head(10)
                display_columns = ['rank_position', 'total_score', 'property_address', 'city', 'equity']
                available_columns = [col for col in display_columns if col in top_10.columns]
                st.dataframe(top_10[available_columns], use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Export failed: {e}")
    
    # Export history
    st.markdown("### 📜 Export History")
    try:
        history_df = exporter.get_export_history(limit=20)
        if not history_df.empty:
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No export history available")
    except Exception as e:
        st.error(f"Error loading export history: {e}")

# ==============================================
# MAIN APPLICATION
# ==============================================
def main():
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Status")
    data_summary = load_data_summary()
    st.sidebar.metric("Properties", data_summary['properties'])
    st.sidebar.metric("Permits", data_summary['permits'])
    st.sidebar.metric("Legal Events", data_summary['legal_events'])
    
    # Page routing
    if page == "📤 Upload & Validate":
        upload_validation_page()
    elif page == "🎯 Scoring Engine":
        scoring_engine_page()
    elif page == "🔧 Weight Tuning":
        weight_tuning_page()
    elif page == "📊 Backtesting":
        backtesting_page()
    elif page == "📁 Export Results":
        export_results_page()

if __name__ == "__main__":
    main()
