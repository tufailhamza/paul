"""
Scoring Engine for Augur Seller Scoring
Implements rule-based scoring with time decay and configurable weights
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sqlalchemy import create_engine, text
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

class ScoringEngine:
    def __init__(self, db_engine=None):
        """Initialize the scoring engine"""
        self.db_engine = db_engine or self._create_db_engine()
        self.weights = {
            'tenure': 0.30,
            'equity': 0.25,
            'legal': 0.20,
            'permit': 0.15,
            'listing': 0.10,
            'maintenance': 0.00
        }
        self.time_decay_half_life_days = 90
        
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
    
    def set_weights(self, weights: Dict[str, float]):
        """Update scoring weights"""
        self.weights.update(weights)
        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def set_time_decay(self, half_life_days: int):
        """Set time decay half-life in days"""
        self.time_decay_half_life_days = half_life_days
    
    def calculate_time_decay(self, event_date, current_date: datetime = None) -> float:
        """Calculate time decay factor for an event"""
        if current_date is None:
            current_date = datetime.now()
        
        if pd.isna(event_date):
            return 0.0
        
        # Convert event_date to datetime if needed
        try:
            if hasattr(event_date, 'date') and not hasattr(event_date, 'hour'):
                # It's a date object, convert to datetime
                event_date = datetime.combine(event_date, datetime.min.time())
            elif isinstance(event_date, str):
                # It's a string, try to parse it
                event_date = pd.to_datetime(event_date)
            elif hasattr(event_date, 'to_pydatetime'):
                # It's a pandas timestamp, convert to datetime
                event_date = event_date.to_pydatetime()
            
            # Ensure both are datetime objects
            if not isinstance(event_date, datetime):
                event_date = pd.to_datetime(event_date)
                if hasattr(event_date, 'to_pydatetime'):
                    event_date = event_date.to_pydatetime()
                else:
                    event_date = datetime.combine(event_date.date(), datetime.min.time())
                    
        except Exception as e:
            print(f"DEBUG: Error converting event_date {event_date} (type: {type(event_date)}): {e}")
            return 0.0
            
        days_ago = (current_date - event_date).days
        if days_ago < 0:
            return 1.0  # Future events get full weight
            
        # Exponential decay: weight = 2^(-days/half_life)
        decay_factor = 2 ** (-days_ago / self.time_decay_half_life_days)
        return max(0.0, min(1.0, decay_factor))
    
    def calculate_tenure_score(self, properties_df: pd.DataFrame) -> pd.Series:
        """Calculate tenure-based selling likelihood score using actual ownership data"""
        print(f"DEBUG: calculate_tenure_score - Input columns: {list(properties_df.columns)}")
        
        scores = pd.Series(0.0, index=properties_df.index)
        
        # Use Last Sale Date to calculate actual ownership tenure
        if 'last_sale_date' in properties_df.columns:
            print(f"DEBUG: calculate_tenure_score - Using last_sale_date for tenure calculation")
            current_date = datetime.now().date()
            
            for idx, row in properties_df.iterrows():
                last_sale_date = row.get('last_sale_date')
                if pd.notna(last_sale_date) and last_sale_date is not None:
                    try:
                        # Calculate years of ownership
                        if isinstance(last_sale_date, str):
                            last_sale_date = pd.to_datetime(last_sale_date).date()
                        
                        years_owned = (current_date - last_sale_date).days / 365.25
                        
                        # Higher tenure = higher likelihood to sell
                        # Scale: 0-5 years = 0.2-0.8, 5+ years = 0.8-1.0
                        if years_owned <= 5:
                            tenure_score = 0.2 + (years_owned / 5) * 0.6
                        else:
                            tenure_score = 0.8 + min(0.2, (years_owned - 5) / 10)
                        
                        # Use .loc instead of .iloc to avoid index issues
                        scores.loc[idx] = min(1.0, tenure_score)
                        # print(f"DEBUG: Property {idx}: {years_owned:.1f} years owned, score: {scores.loc[idx]:.3f}")
                    except Exception as e:
                        print(f"DEBUG: Error calculating tenure for property {idx}: {e}")
                        scores.loc[idx] = 0.5  # Default score
                else:
                    scores.loc[idx] = 0.5  # Default score for missing data
        else:
            # Fallback: use equity as proxy for tenure
            if 'equity' in properties_df.columns:
                print(f"DEBUG: calculate_tenure_score - Using equity as proxy for tenure")
                equity_scores = properties_df['equity'].fillna(0)
                # Higher equity suggests longer ownership
                scores = equity_scores / 1000000.0  # Normalize large equity values
                scores = scores.clip(0, 1)  # Cap at 1.0
            else:
                # Default scoring
                scores = pd.Series(0.5, index=properties_df.index)
                print(f"DEBUG: calculate_tenure_score - Using default 0.5 scores")
        
        print(f"DEBUG: calculate_tenure_score - Final tenure scores: {scores.head().tolist()}")
        return scores
    
    def calculate_equity_score(self, properties_df: pd.DataFrame) -> pd.Series:
        """Calculate equity-based selling likelihood score using actual equity values"""
        print(f"DEBUG: calculate_equity_score - Equity column exists: {'equity' in properties_df.columns}")
        if 'equity' not in properties_df.columns:
            print("DEBUG: calculate_equity_score - No equity column, returning zeros")
            return pd.Series(0.0, index=properties_df.index)
        
        equity = properties_df['equity'].fillna(0)
        print(f"DEBUG: calculate_equity_score - Equity values: {equity.head().tolist()}")
        print(f"DEBUG: calculate_equity_score - Equity range: {equity.min()} to {equity.max()}")
        
        # Higher equity = higher likelihood to sell (more profit potential)
        # Use percentile-based scoring for better distribution
        scores = pd.Series(0.0, index=properties_df.index)
        
        # Calculate percentiles for normalization
        equity_clean = equity[equity > 0]  # Only positive equity values
        if len(equity_clean) > 0:
            p25 = equity_clean.quantile(0.25)
            p50 = equity_clean.quantile(0.50)
            p75 = equity_clean.quantile(0.75)
            p90 = equity_clean.quantile(0.90)
            
            print(f"DEBUG: calculate_equity_score - Equity percentiles: 25%={p25:.0f}, 50%={p50:.0f}, 75%={p75:.0f}, 90%={p90:.0f}")
            
            for idx, eq_val in equity.items():
                if eq_val <= 0:
                    scores.loc[idx] = 0.0  # No equity = no score
                elif eq_val <= p25:
                    scores.loc[idx] = 0.2  # Low equity
                elif eq_val <= p50:
                    scores.loc[idx] = 0.4  # Below median
                elif eq_val <= p75:
                    scores.loc[idx] = 0.6  # Above median
                elif eq_val <= p90:
                    scores.loc[idx] = 0.8  # High equity
                else:
                    scores.loc[idx] = 1.0  # Very high equity
        else:
            # No positive equity values, use default
            scores = pd.Series(0.5, index=properties_df.index)
        
        print(f"DEBUG: calculate_equity_score - Final equity scores: {scores.head().tolist()}")
        return scores
    
    def calculate_legal_score(self, properties_df: pd.DataFrame) -> pd.Series:
        """Calculate legal events score (probate, divorce, lis pendens)"""
        scores = pd.Series(0.0, index=properties_df.index)
        
        # Get legal events for properties
        tmk_apns = properties_df['tmk_apn'].dropna().unique()
        if len(tmk_apns) == 0:
            return scores
        
        # Query legal events
        query = """
        SELECT tmk_apn, case_type, filing_date, status
        FROM legal_events 
        WHERE tmk_apn = ANY(:tmk_apns)
        AND status IN ('Open', 'Pending')
        """
        
        try:
            with self.db_engine.connect() as conn:
                legal_df = pd.read_sql(
                    text(query), 
                    conn, 
                    params={'tmk_apns': list(tmk_apns)}
                )
        except Exception as e:
            print(f"Error querying legal events: {e}")
            return scores
        
        if legal_df.empty:
            return scores
        
        # Calculate scores based on legal events
        for idx, row in properties_df.iterrows():
            tmk_apn = row.get('tmk_apn')
            if pd.isna(tmk_apn):
                continue
                
            property_legal = legal_df[legal_df['tmk_apn'] == tmk_apn]
            if property_legal.empty:
                continue
            
            # Different case types have different weights
            case_weights = {
                'Probate': 0.9,
                'Divorce': 0.8,
                'Lis Pendens': 0.7,
                'Foreclosure': 0.6,
                'Bankruptcy': 0.5
            }
            
            max_score = 0.0
            for _, legal_event in property_legal.iterrows():
                case_type = legal_event.get('case_type', 'Other')
                filing_date = legal_event.get('filing_date')
                
                # Apply time decay
                decay_factor = self.calculate_time_decay(filing_date)
                case_score = case_weights.get(case_type, 0.3) * decay_factor
                max_score = max(max_score, case_score)
            
            scores.loc[idx] = max_score
        
        return scores
    
    def calculate_permit_score(self, properties_df: pd.DataFrame) -> pd.Series:
        """Calculate permit activity score"""
        scores = pd.Series(0.0, index=properties_df.index)
        
        # Get permits for properties
        addresses = properties_df['address'].dropna().unique()
        if len(addresses) == 0:
            return scores
        
        # Query permits (using address matching)
        query = """
        SELECT address, issued_date, status, permit_type
        FROM permits 
        WHERE address = ANY(:addresses)
        AND issued_date >= :cutoff_date
        """
        
        cutoff_date = datetime.now() - timedelta(days=365)  # Last year
        
        try:
            with self.db_engine.connect() as conn:
                permits_df = pd.read_sql(
                    text(query), 
                    conn, 
                    params={'addresses': list(addresses), 'cutoff_date': cutoff_date}
                )
        except Exception as e:
            print(f"Error querying permits: {e}")
            return scores
        
        if permits_df.empty:
            return scores
        
        # Calculate scores based on permit activity
        for idx, row in properties_df.iterrows():
            address = row.get('address')
            if pd.isna(address):
                continue
                
            property_permits = permits_df[permits_df['address'].str.contains(address, case=False, na=False)]
            if property_permits.empty:
                continue
            
            # Recent permit activity suggests potential for sale
            total_score = 0.0
            for _, permit in property_permits.iterrows():
                issued_date = permit.get('issued_date')
                decay_factor = self.calculate_time_decay(issued_date)
                
                # Different permit types have different weights
                permit_weights = {
                    'Building Permit': 0.8,
                    'Renovation': 0.7,
                    'Addition': 0.6,
                    'Repair': 0.4
                }
                
                permit_type = permit.get('permit_type', '')
                weight = 0.5  # Default
                for ptype, pweight in permit_weights.items():
                    if ptype.lower() in permit_type.lower():
                        weight = pweight
                        break
                
                total_score += weight * decay_factor
            
            # Normalize score
            scores.loc[idx] = min(1.0, total_score / 3.0)  # Cap at 1.0
        
        return scores
    
    def calculate_listing_score(self, properties_df: pd.DataFrame) -> pd.Series:
        """Calculate nearby listing activity score"""
        scores = pd.Series(0.0, index=properties_df.index)
        
        # Get recent listings in the same area
        query = """
        SELECT address, district, report_date, status
        FROM properties 
        WHERE status IN ('Active', 'Sold', 'ACT', 'SLD')
        AND report_date >= :cutoff_date
        AND district = ANY(:districts)
        """
        
        cutoff_date = datetime.now() - timedelta(days=180)  # Last 6 months
        districts = properties_df['district'].dropna().unique()
        
        if len(districts) == 0:
            return scores
        
        try:
            with self.db_engine.connect() as conn:
                listings_df = pd.read_sql(
                    text(query), 
                    conn, 
                    params={'cutoff_date': cutoff_date, 'districts': list(districts)}
                )
        except Exception as e:
            print(f"Error querying listings: {e}")
            return scores
        
        if listings_df.empty:
            return scores
        
        # Calculate scores based on nearby listing activity
        for idx, row in properties_df.iterrows():
            district = row.get('district')
            if pd.isna(district):
                continue
                
            district_listings = listings_df[listings_df['district'] == district]
            if district_listings.empty:
                continue
            
            # More recent listings in the area = higher score
            total_score = 0.0
            for _, listing in district_listings.iterrows():
                report_date = listing.get('report_date')
                decay_factor = self.calculate_time_decay(report_date)
                total_score += decay_factor
            
            # Normalize by number of properties in district
            scores.loc[idx] = min(1.0, total_score / 10.0)  # Cap at 1.0
        
        return scores
    
    def calculate_maintenance_score(self, properties_df: pd.DataFrame) -> pd.Series:
        """Calculate maintenance burden score using actual property data"""
        print(f"DEBUG: calculate_maintenance_score - Available columns: {[col for col in properties_df.columns if col in ['living_area', 'year_built', 'bedrooms', 'property_type_detail']]}")
        
        scores = pd.Series(0.0, index=properties_df.index)
        
        for idx, row in properties_df.iterrows():
            score = 0.0
            
            # Factor 1: Property size (larger = more maintenance)
            if 'living_area' in row and pd.notna(row['living_area']):
                living_area = float(row['living_area']) if row['living_area'] else 0
                if living_area > 0:
                    # Normalize: 0-3000 sqft = 0-0.3 score
                    size_score = min(0.3, living_area / 3000)
                    score += size_score
                    # print(f"DEBUG: Property {idx}: {living_area} sqft -> {size_score:.3f}")
            
            # Factor 2: Property age (older = more maintenance)
            if 'year_built' in row and pd.notna(row['year_built']):
                try:
                    year_built = int(row['year_built'])
                    current_year = datetime.now().year
                    age = current_year - year_built
                    if age > 0:
                        # Normalize: 0-50 years = 0-0.4 score
                        age_score = min(0.4, age / 50)
                        score += age_score
                        # print(f"DEBUG: Property {idx}: {age} years old -> {age_score:.3f}")
                except Exception as e:
                    print(f"DEBUG: Error calculating age for property {idx}: {e}")
            
            # Factor 3: Stairs detection (address contains indicators)
            if 'address' in row and pd.notna(row['address']):
                address = str(row['address']).lower()
                stairs_indicators = ['stairs', 'stair', 'level', 'floor', 'upstairs', 'downstairs', 'apt', 'unit']
                if any(indicator in address for indicator in stairs_indicators):
                    score += 0.2  # Stairs add maintenance burden
                    # print(f"DEBUG: Property {idx}: Stairs detected in address")
            
            # Factor 4: Property type (condos/townhouses = less maintenance)
            if 'property_type_detail' in row and pd.notna(row['property_type_detail']):
                prop_type = str(row['property_type_detail']).lower()
                if 'condo' in prop_type or 'townhouse' in prop_type or 'apartment' in prop_type:
                    score -= 0.1  # Reduce maintenance burden for condos
                    # print(f"DEBUG: Property {idx}: Condo/Apartment -> reduced maintenance")
                elif 'single family' in prop_type or 'sf' in prop_type:
                    score += 0.1  # Increase for single family homes
                    # print(f"DEBUG: Property {idx}: Single Family -> increased maintenance")
            
            # Factor 5: Bedrooms (more bedrooms = more maintenance)
            if 'bedrooms' in row and pd.notna(row['bedrooms']):
                try:
                    bedrooms = int(row['bedrooms'])
                    if bedrooms > 0:
                        # Normalize: 1-5 bedrooms = 0-0.2 score
                        bedroom_score = min(0.2, (bedrooms - 1) / 4)
                        score += bedroom_score
                        # print(f"DEBUG: Property {idx}: {bedrooms} bedrooms -> {bedroom_score:.3f}")
                except Exception as e:
                    print(f"DEBUG: Error calculating bedroom score for property {idx}: {e}")
            
            # Factor 6: Owner occupied (rentals = more maintenance)
            if 'owner_occupied' in row and pd.notna(row['owner_occupied']):
                if str(row['owner_occupied']).lower() == 'no':
                    score += 0.1  # Rental properties need more maintenance
                    # print(f"DEBUG: Property {idx}: Not owner occupied -> increased maintenance")
            
            scores.loc[idx] = min(1.0, max(0.0, score))
            # print(f"DEBUG: Property {idx}: Total maintenance score = {scores.loc[idx]:.3f}")
        
        print(f"DEBUG: calculate_maintenance_score - Final scores: {scores.head().tolist()}")
        return scores
    
    def calculate_total_scores(self, properties_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate total scores for all properties"""
        print(f"DEBUG: Starting score calculation for {len(properties_df)} properties")
        print(f"DEBUG: Available columns: {list(properties_df.columns)}")
        print(f"DEBUG: Sample property data:")
        print(properties_df.head(2).to_dict('records'))
        
        # Calculate individual component scores
        print("DEBUG: Calculating tenure scores...")
        tenure_scores = self.calculate_tenure_score(properties_df)
        print(f"DEBUG: Tenure scores range: {tenure_scores.min():.3f} - {tenure_scores.max():.3f}")
        print(f"DEBUG: Tenure scores sample: {tenure_scores.head().tolist()}")
        
        print("DEBUG: Calculating equity scores...")
        equity_scores = self.calculate_equity_score(properties_df)
        print(f"DEBUG: Equity scores range: {equity_scores.min():.3f} - {equity_scores.max():.3f}")
        print(f"DEBUG: Equity scores sample: {equity_scores.head().tolist()}")
        
        print("DEBUG: Calculating legal scores...")
        legal_scores = self.calculate_legal_score(properties_df)
        print(f"DEBUG: Legal scores range: {legal_scores.min():.3f} - {legal_scores.max():.3f}")
        print(f"DEBUG: Legal scores sample: {legal_scores.head().tolist()}")
        
        print("DEBUG: Calculating permit scores...")
        permit_scores = self.calculate_permit_score(properties_df)
        print(f"DEBUG: Permit scores range: {permit_scores.min():.3f} - {permit_scores.max():.3f}")
        print(f"DEBUG: Permit scores sample: {permit_scores.head().tolist()}")
        
        print("DEBUG: Calculating listing scores...")
        listing_scores = self.calculate_listing_score(properties_df)
        print(f"DEBUG: Listing scores range: {listing_scores.min():.3f} - {listing_scores.max():.3f}")
        print(f"DEBUG: Listing scores sample: {listing_scores.head().tolist()}")
        
        print("DEBUG: Calculating maintenance scores...")
        maintenance_scores = self.calculate_maintenance_score(properties_df)
        print(f"DEBUG: Maintenance scores range: {maintenance_scores.min():.3f} - {maintenance_scores.max():.3f}")
        print(f"DEBUG: Maintenance scores sample: {maintenance_scores.head().tolist()}")
        
        print(f"DEBUG: Weights being used: {self.weights}")
        
        # Calculate weighted total score
        total_scores = (
            tenure_scores * self.weights['tenure'] +
            equity_scores * self.weights['equity'] +
            legal_scores * self.weights['legal'] +
            permit_scores * self.weights['permit'] +
            listing_scores * self.weights['listing'] +
            maintenance_scores * self.weights['maintenance']
        )
        
        print(f"DEBUG: Total scores range: {total_scores.min():.3f} - {total_scores.max():.3f}")
        print(f"DEBUG: Total scores sample: {total_scores.head().tolist()}")
        
        # Create results dataframe
        results_df = properties_df.copy()
        results_df['tenure_score'] = tenure_scores
        results_df['equity_score'] = equity_scores
        results_df['legal_score'] = legal_scores
        results_df['permit_score'] = permit_scores
        results_df['listing_score'] = listing_scores
        results_df['maintenance_score'] = maintenance_scores
        results_df['total_score'] = total_scores
        
        # Add ranking
        results_df['rank_position'] = results_df['total_score'].rank(ascending=False, method='dense').astype(int)
        
        # Debug: Print unique ranks
        unique_ranks = sorted(results_df['rank_position'].unique())
        print(f"DEBUG: Unique ranks found: {unique_ranks}")
        print(f"DEBUG: Total unique ranks: {len(unique_ranks)}")
        
        print(f"DEBUG: Final results sample:")
        print(results_df[['address', 'city', 'equity', 'tenure_score', 'equity_score', 'legal_score', 'total_score', 'rank_position']].head())
        
        return results_df
    
    def save_scores_to_db(self, scores_df: pd.DataFrame, model_version: str = "v1.0"):
        """Save scoring results to database"""
        # Prepare data for insertion
        scoring_data = []
        for _, row in scores_df.iterrows():
            scoring_data.append({
                'property_id': row.get('id'),
                'tmk_apn': row.get('tmk_apn'),
                'address': row.get('address'),
                'score': row.get('total_score'),
                'tenure_score': row.get('tenure_score'),
                'equity_score': row.get('equity_score'),
                'legal_score': row.get('legal_score'),
                'permit_score': row.get('permit_score'),
                'listing_score': row.get('listing_score'),
                'maintenance_score': row.get('maintenance_score'),
                'total_score': row.get('total_score'),
                'rank_position': row.get('rank_position'),
                'model_version': model_version,
                'weights_config': json.dumps(self.weights)
            })
        
        # Insert into database
        insert_query = """
        INSERT INTO scoring_results 
        (property_id, tmk_apn, address, score, tenure_score, equity_score, 
         legal_score, permit_score, listing_score, maintenance_score, 
         total_score, rank_position, model_version, weights_config)
        VALUES 
        (:property_id, :tmk_apn, :address, :score, :tenure_score, :equity_score,
         :legal_score, :permit_score, :listing_score, :maintenance_score,
         :total_score, :rank_position, :model_version, :weights_config)
        ON CONFLICT (property_id, model_version) 
        DO UPDATE SET
            tmk_apn = EXCLUDED.tmk_apn,
            address = EXCLUDED.address,
            score = EXCLUDED.score,
            tenure_score = EXCLUDED.tenure_score,
            equity_score = EXCLUDED.equity_score,
            legal_score = EXCLUDED.legal_score,
            permit_score = EXCLUDED.permit_score,
            listing_score = EXCLUDED.listing_score,
            maintenance_score = EXCLUDED.maintenance_score,
            total_score = EXCLUDED.total_score,
            rank_position = EXCLUDED.rank_position,
            weights_config = EXCLUDED.weights_config
        """
        
        try:
            with self.db_engine.begin() as conn:
                conn.execute(text(insert_query), scoring_data)
            return True
        except Exception as e:
            print(f"Error saving scores to database: {e}")
            return False
