"""
Export functionality for Augur Seller Scoring
Handles Top-K CSV and GHL dry-run CSV exports
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from sqlalchemy import text, create_engine
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

class Exporter:
    def __init__(self, db_engine=None):
        """Initialize the exporter"""
        self.db_engine = db_engine or self._create_db_engine()
        
    def _create_db_engine(self):
        """Create database engine from environment variables"""
        # Use st.secrets for deployment compatibility
        try:
            db_user = st.secrets["DB_USER"]
            db_password = st.secrets["DB_PASSWORD"]
            db_host = st.secrets["DB_HOST"]
            db_name = st.secrets["DB_NAME"]
            db_port = st.secrets["DB_PORT"]
        except:
            # Fallback to os.getenv for local development
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")
            db_host = os.getenv("DB_HOST")
            db_name = os.getenv("DB_NAME")
            db_port = os.getenv("DB_PORT", "5432")
        
        return create_engine(
            f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )
    
    def export_top_k_csv(self, top_k_df: pd.DataFrame, k: int = 100, 
                        export_path: str = None) -> str:
        """Export top K properties to CSV format"""
        
        if export_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"exports/top_k_{k}_{timestamp}.csv"
        
        # Create exports directory if it doesn't exist
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        
        # Prepare data for export
        export_df = top_k_df.copy()
        
        # Select relevant columns for export
        export_columns = [
            'rank_position', 'total_score', 'address', 'city', 'state', 'zip_code',
            'owner_name', 'tmk_apn', 'phone', 'email', 'equity', 'bedrooms', 'bathrooms',
            'living_area', 'list_price', 'tenure_score', 'equity_score', 'legal_score',
            'permit_score', 'listing_score', 'maintenance_score'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in export_columns if col in export_df.columns]
        export_df = export_df[available_columns]
        
        # Rename columns for better readability
        column_mapping = {
            'rank_position': 'Rank',
            'total_score': 'Seller_Score',
            'address': 'Address',
            'city': 'City',
            'state': 'State',
            'zip_code': 'ZIP',
            'owner_name': 'Owner_Name',
            'tmk_apn': 'TMK_APN',
            'phone': 'Phone',
            'email': 'Email',
            'equity': 'Equity_Percent',
            'bedrooms': 'Bedrooms',
            'bathrooms': 'Bathrooms',
            'living_area': 'Living_Area_SF',
            'list_price': 'List_Price',
            'tenure_score': 'Tenure_Score',
            'equity_score': 'Equity_Score',
            'legal_score': 'Legal_Score',
            'permit_score': 'Permit_Score',
            'listing_score': 'Listing_Score',
            'maintenance_score': 'Maintenance_Score'
        }
        
        export_df = export_df.rename(columns=column_mapping)
        
        # Format numeric columns
        if 'Seller_Score' in export_df.columns:
            export_df['Seller_Score'] = export_df['Seller_Score'].round(4)
        
        if 'Equity_Percent' in export_df.columns:
            export_df['Equity_Percent'] = export_df['Equity_Percent'].round(2)
        
        if 'List_Price' in export_df.columns:
            export_df['List_Price'] = export_df['List_Price'].apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) and x > 0 else ""
            )
        
        # Format score columns
        score_columns = ['Tenure_Score', 'Equity_Score', 'Legal_Score', 
                        'Permit_Score', 'Listing_Score', 'Maintenance_Score']
        for col in score_columns:
            if col in export_df.columns:
                export_df[col] = export_df[col].round(4)
        
        # Sort by rank
        if 'Rank' in export_df.columns:
            export_df = export_df.sort_values('Rank')
        
        # Export to CSV
        export_df.to_csv(export_path, index=False)
        
        # Log export
        self._log_export('top_k_csv', len(export_df), export_path)
        
        return export_path
    
    def export_ghl_dry_run_csv(self, top_k_df: pd.DataFrame, k: int = 100,
                              export_path: str = None) -> str:
        """Export top K properties in GHL-ready format"""
        
        if export_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"exports/ghl_dry_run_{k}_{timestamp}.csv"
        
        # Create exports directory if it doesn't exist
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        
        # Prepare data for GHL format
        ghl_df = top_k_df.copy()
        
        # GHL format columns (adjust based on your GHL setup)
        ghl_columns = {
            'owner_name': 'Contact Name',
            'address': 'Address',
            'city': 'City',
            'state': 'State',
            'zip_code': 'ZIP Code',
            'phone': 'Phone',
            'email': 'Email',
            'total_score': 'Seller Score',
            'equity': 'Equity %',
            'tmk_apn': 'Property ID'
        }
        
        # Create GHL dataframe
        ghl_export_df = pd.DataFrame()
        
        for source_col, target_col in ghl_columns.items():
            if source_col in ghl_df.columns:
                ghl_export_df[target_col] = ghl_df[source_col]
            else:
                ghl_export_df[target_col] = ""
        
        # Add GHL-specific columns
        ghl_export_df['Lead Source'] = 'Augur Seller Scoring'
        ghl_export_df['Lead Status'] = 'New'
        ghl_export_df['Lead Score'] = ghl_export_df['Seller Score'].round(2)
        ghl_export_df['Notes'] = f"High probability seller - Score: {ghl_export_df['Seller Score'].round(2)}"
        
        # Format phone numbers for GHL
        if 'Phone' in ghl_export_df.columns:
            ghl_export_df['Phone'] = ghl_export_df['Phone'].apply(self._format_phone_for_ghl)
        
        # Format equity percentage
        if 'Equity %' in ghl_export_df.columns:
            ghl_export_df['Equity %'] = ghl_export_df['Equity %'].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) and x > 0 else ""
            )
        
        # Sort by seller score (highest first)
        if 'Seller Score' in ghl_export_df.columns:
            ghl_export_df = ghl_export_df.sort_values('Seller Score', ascending=False)
        
        # Export to CSV
        ghl_export_df.to_csv(export_path, index=False)
        
        # Log export
        self._log_export('ghl_dry_run', len(ghl_export_df), export_path)
        
        return export_path
    
    def _format_phone_for_ghl(self, phone: str) -> str:
        """Format phone number for GHL"""
        if pd.isna(phone) or phone == "":
            return ""
        
        # Remove all non-digit characters
        digits = ''.join(filter(str.isdigit, str(phone)))
        
        # Format as (XXX) XXX-XXXX
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        else:
            return str(phone)  # Return original if can't format
    
    def _log_export(self, export_type: str, record_count: int, file_path: str):
        """Log export activity to database"""
        insert_query = """
        INSERT INTO export_logs 
        (export_type, record_count, file_path, export_date)
        VALUES 
        (:export_type, :record_count, :file_path, :export_date)
        """
        
        try:
            with self.db_engine.begin() as conn:
                conn.execute(text(insert_query), {
                    'export_type': export_type,
                    'record_count': record_count,
                    'file_path': file_path,
                    'export_date': datetime.now()
                })
        except Exception as e:
            print(f"Error logging export: {e}")
    
    def get_export_history(self, limit: int = 50) -> pd.DataFrame:
        """Get export history from database"""
        query = """
        SELECT export_type, record_count, file_path, export_date
        FROM export_logs
        ORDER BY export_date DESC
        LIMIT :limit
        """
        
        try:
            with self.db_engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params={'limit': limit})
        except Exception as e:
            print(f"Error getting export history: {e}")
            return pd.DataFrame()
        
        return df
    
    def get_top_k_properties(self, k: int = 100, weights: Dict = None, 
                           time_decay_half_life: int = 90) -> pd.DataFrame:
        """Get top K scoring properties for export"""
        
        # Import here to avoid circular imports
        from scoring.engine import ScoringEngine
        
        # Create scoring engine
        scoring_engine = ScoringEngine(self.db_engine)
        
        # Set scoring parameters
        if weights:
            scoring_engine.set_weights(weights)
        scoring_engine.set_time_decay(time_decay_half_life)
        
        # Get all properties
        query = """
        SELECT * FROM properties 
        WHERE status IN ('Active', 'Sold', 'Off Market', 'Pending', 'ACT', 'PND') OR status IS NULL
        ORDER BY report_date DESC
        """
        
        try:
            with self.db_engine.connect() as conn:
                df = pd.read_sql(query, conn)
        except Exception as e:
            print(f"Error getting properties: {e}")
            return pd.DataFrame()
        
        if df.empty:
            return pd.DataFrame()
        
        # Calculate scores
        try:
            scores_df = scoring_engine.calculate_total_scores(df)
            # Get top K
            top_k_df = scores_df.nlargest(k, 'total_score')
            return top_k_df
        except Exception as e:
            print(f"Error calculating scores: {e}")
            return pd.DataFrame()
    
    def create_export_summary(self, top_k_df: pd.DataFrame) -> Dict:
        """Create summary statistics for export"""
        summary = {
            'total_records': len(top_k_df),
            'export_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'score_range': {
                'min': float(top_k_df['total_score'].min()) if 'total_score' in top_k_df.columns else 0.0,
                'max': float(top_k_df['total_score'].max()) if 'total_score' in top_k_df.columns else 0.0,
                'mean': float(top_k_df['total_score'].mean()) if 'total_score' in top_k_df.columns else 0.0
            }
        }
        
        # Add equity statistics
        if 'equity' in top_k_df.columns:
            equity_data = top_k_df['equity'].dropna()
            if len(equity_data) > 0:
                summary['equity_stats'] = {
                    'min': float(equity_data.min()),
                    'max': float(equity_data.max()),
                    'mean': float(equity_data.mean()),
                    'count_with_equity': len(equity_data)
                }
        
        # Add contact information statistics
        if 'phone' in top_k_df.columns:
            phone_count = top_k_df['phone'].notna().sum()
            summary['contact_stats'] = {
                'phone_count': int(phone_count),
                'phone_percentage': float(phone_count / len(top_k_df) * 100)
            }
        
        if 'email' in top_k_df.columns:
            email_count = top_k_df['email'].notna().sum()
            if 'contact_stats' not in summary:
                summary['contact_stats'] = {}
            summary['contact_stats']['email_count'] = int(email_count)
            summary['contact_stats']['email_percentage'] = float(email_count / len(top_k_df) * 100)
        
        return summary
    
    def validate_export_data(self, df: pd.DataFrame) -> Dict:
        """Validate data before export"""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check for required columns
        required_columns = ['address', 'total_score']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
            validation_results['is_valid'] = False
        
        # Check for empty dataframe
        if len(df) == 0:
            validation_results['errors'].append("No data to export")
            validation_results['is_valid'] = False
        
        # Check for duplicate addresses
        if 'address' in df.columns:
            duplicate_addresses = df['address'].duplicated().sum()
            if duplicate_addresses > 0:
                validation_results['warnings'].append(f"{duplicate_addresses} duplicate addresses found")
        
        # Check for missing contact information
        if 'phone' in df.columns and 'email' in df.columns:
            no_contact = df['phone'].isna() & df['email'].isna()
            no_contact_count = no_contact.sum()
            if no_contact_count > 0:
                validation_results['warnings'].append(f"{no_contact_count} records have no phone or email")
        
        # Check score distribution
        if 'total_score' in df.columns:
            score_stats = df['total_score'].describe()
            if score_stats['min'] < 0 or score_stats['max'] > 1:
                validation_results['warnings'].append("Scores outside expected range (0-1)")
        
        return validation_results
