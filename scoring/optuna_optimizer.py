"""
Optuna Hyperparameter Optimization for Augur Seller Scoring
Implements time-aware cross-validation and precision@K optimization
"""

import optuna
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

from .engine import ScoringEngine

class OptunaOptimizer:
    def __init__(self, db_engine, n_trials: int = 200, timeout_seconds: int = 3600):
        """Initialize the Optuna optimizer"""
        self.db_engine = db_engine
        self.n_trials = n_trials
        self.timeout_seconds = timeout_seconds
        self.scoring_engine = ScoringEngine(db_engine)
        
    def get_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get training data with labels for optimization"""
        # Get properties with known outcomes (sold properties)
        query = """
        SELECT p.*, 
               CASE WHEN p.status = 'Sold' THEN 1 ELSE 0 END as sold_label,
               p.report_date as outcome_date
        FROM properties p
        WHERE p.status IN ('Sold', 'Off Market', 'Active', 'Expired')
        AND p.report_date IS NOT NULL
        ORDER BY p.report_date
        """
        
        try:
            with self.db_engine.connect() as conn:
                df = pd.read_sql(query, conn)
        except Exception as e:
            print(f"Error getting training data: {e}")
            return pd.DataFrame(), pd.Series()
        
        if df.empty:
            return pd.DataFrame(), pd.Series()
        
        # Separate features and labels
        feature_columns = [col for col in df.columns if col not in ['sold_label', 'outcome_date', 'id']]
        X = df[feature_columns]
        y = df['sold_label']
        
        return X, y
    
    def create_time_aware_splits(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> List[Tuple]:
        """Create time-aware splits for cross-validation"""
        # Sort by date
        if 'report_date' in X.columns:
            date_col = 'report_date'
        else:
            # If no date column, use index as proxy
            date_col = None
            X = X.copy()
            X['date_proxy'] = range(len(X))
            date_col = 'date_proxy'
        
        # Sort by date
        sorted_indices = X[date_col].argsort()
        X_sorted = X.iloc[sorted_indices]
        y_sorted = y.iloc[sorted_indices]
        
        # Create time-aware splits
        splits = []
        n_samples = len(X_sorted)
        
        for i in range(n_splits):
            # Calculate split points
            train_end = int(n_samples * (i + 1) / (n_splits + 1))
            test_start = train_end
            test_end = int(n_samples * (i + 2) / (n_splits + 1))
            
            if test_end > n_samples:
                test_end = n_samples
            
            if test_start >= test_end:
                continue
            
            train_indices = sorted_indices[:train_end]
            test_indices = sorted_indices[test_start:test_end]
            
            splits.append((train_indices, test_indices))
        
        return splits
    
    def calculate_precision_at_k(self, y_true: np.ndarray, y_scores: np.ndarray, k: int = 100) -> float:
        """Calculate precision@K"""
        if len(y_true) == 0:
            return 0.0
        
        # Get top K predictions
        top_k_indices = np.argsort(y_scores)[-k:]
        top_k_labels = y_true[top_k_indices]
        
        # Calculate precision
        if len(top_k_labels) == 0:
            return 0.0
        
        precision = np.sum(top_k_labels) / len(top_k_labels)
        return precision
    
    def calculate_pr_auc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Calculate Precision-Recall AUC"""
        if len(np.unique(y_true)) < 2:
            return 0.0
        
        try:
            pr_auc = average_precision_score(y_true, y_scores)
            return pr_auc
        except:
            return 0.0
    
    def objective(self, trial) -> float:
        """Optuna objective function"""
        # Suggest hyperparameters
        tenure_weight = trial.suggest_float('tenure_weight', 0.0, 1.0)
        equity_weight = trial.suggest_float('equity_weight', 0.0, 1.0)
        legal_weight = trial.suggest_float('legal_weight', 0.0, 1.0)
        permit_weight = trial.suggest_float('permit_weight', 0.0, 1.0)
        listing_weight = trial.suggest_float('listing_weight', 0.0, 1.0)
        maintenance_weight = trial.suggest_float('maintenance_weight', 0.0, 1.0)
        time_decay_half_life = trial.suggest_int('time_decay_half_life', 30, 365)
        
        # Normalize weights
        total_weight = tenure_weight + equity_weight + legal_weight + permit_weight + listing_weight + maintenance_weight
        if total_weight == 0:
            return 0.0
        
        weights = {
            'tenure': tenure_weight / total_weight,
            'equity': equity_weight / total_weight,
            'legal': legal_weight / total_weight,
            'permit': permit_weight / total_weight,
            'listing': listing_weight / total_weight,
            'maintenance': maintenance_weight / total_weight
        }
        
        # Set weights and time decay
        self.scoring_engine.set_weights(weights)
        self.scoring_engine.set_time_decay(time_decay_half_life)
        
        # Get training data
        X, y = self.get_training_data()
        if X.empty:
            return 0.0
        
        # Create time-aware splits
        splits = self.create_time_aware_splits(X, y, n_splits=3)
        if not splits:
            return 0.0
        
        # Cross-validation
        cv_scores = []
        for train_idx, test_idx in splits:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Calculate scores for test set
            try:
                test_scores_df = self.scoring_engine.calculate_total_scores(X_test)
                y_scores = test_scores_df['total_score'].values
                
                # Calculate metrics
                precision_at_k = self.calculate_precision_at_k(y_test.values, y_scores, k=min(100, len(y_test)))
                pr_auc = self.calculate_pr_auc(y_test.values, y_scores)
                
                # Combined score (weighted average)
                combined_score = 0.7 * precision_at_k + 0.3 * pr_auc
                cv_scores.append(combined_score)
                
            except Exception as e:
                print(f"Error in CV fold: {e}")
                cv_scores.append(0.0)
        
        # Return mean CV score
        mean_score = np.mean(cv_scores) if cv_scores else 0.0
        
        # Store trial results in database
        self.store_trial_result(trial, mean_score, weights, time_decay_half_life)
        
        return mean_score
    
    def store_trial_result(self, trial, objective_value: float, weights: Dict, time_decay_half_life: int):
        """Store trial result in database"""
        insert_query = """
        INSERT INTO optuna_trials 
        (trial_number, objective_value, tenure_weight, equity_weight, legal_weight, 
         permit_weight, listing_weight, maintenance_weight, time_decay_half_life_days, trial_status)
        VALUES 
        (:trial_number, :objective_value, :tenure_weight, :equity_weight, :legal_weight,
         :permit_weight, :listing_weight, :maintenance_weight, :time_decay_half_life_days, :trial_status)
        """
        
        try:
            with self.db_engine.begin() as conn:
                conn.execute(text(insert_query), {
                    'trial_number': trial.number,
                    'objective_value': objective_value,
                    'tenure_weight': weights['tenure'],
                    'equity_weight': weights['equity'],
                    'legal_weight': weights['legal'],
                    'permit_weight': weights['permit'],
                    'listing_weight': weights['listing'],
                    'maintenance_weight': weights['maintenance'],
                    'time_decay_half_life_days': time_decay_half_life,
                    'trial_status': 'COMPLETE'
                })
        except Exception as e:
            print(f"Error storing trial result: {e}")
    
    def optimize(self) -> Dict:
        """Run Optuna optimization"""
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout_seconds,
            show_progress_bar=True
        )
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        # Normalize best weights
        total_weight = sum([
            best_params['tenure_weight'],
            best_params['equity_weight'],
            best_params['legal_weight'],
            best_params['permit_weight'],
            best_params['listing_weight'],
            best_params['maintenance_weight']
        ])
        
        if total_weight > 0:
            best_weights = {
                'tenure': best_params['tenure_weight'] / total_weight,
                'equity': best_params['equity_weight'] / total_weight,
                'legal': best_params['legal_weight'] / total_weight,
                'permit': best_params['permit_weight'] / total_weight,
                'listing': best_params['listing_weight'] / total_weight,
                'maintenance': best_params['maintenance_weight'] / total_weight
            }
        else:
            best_weights = {
                'tenure': 0.3, 'equity': 0.25, 'legal': 0.2,
                'permit': 0.15, 'listing': 0.1, 'maintenance': 0.0
            }
        
        # Store best configuration
        self.store_best_config(best_weights, best_params['time_decay_half_life'], best_score)
        
        return {
            'best_weights': best_weights,
            'best_time_decay': best_params['time_decay_half_life'],
            'best_score': best_score,
            'n_trials': len(study.trials),
            'study': study
        }
    
    def store_best_config(self, weights: Dict, time_decay: int, score: float):
        """Store best configuration in database"""
        # Deactivate current active config
        deactivate_query = "UPDATE model_configurations SET is_active = FALSE"
        
        # Insert new best config
        insert_query = """
        INSERT INTO model_configurations 
        (config_name, tenure_weight, equity_weight, legal_weight, permit_weight, 
         listing_weight, maintenance_weight, time_decay_half_life_days, is_active, performance_metrics)
        VALUES 
        (:config_name, :tenure_weight, :equity_weight, :legal_weight, :permit_weight,
         :listing_weight, :maintenance_weight, :time_decay_half_life_days, :is_active, :performance_metrics)
        """
        
        try:
            with self.db_engine.begin() as conn:
                conn.execute(text(deactivate_query))
                conn.execute(text(insert_query), {
                    'config_name': f'optuna_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                    'tenure_weight': weights['tenure'],
                    'equity_weight': weights['equity'],
                    'legal_weight': weights['legal'],
                    'permit_weight': weights['permit'],
                    'listing_weight': weights['listing'],
                    'maintenance_weight': weights['maintenance'],
                    'time_decay_half_life_days': time_decay,
                    'is_active': True,
                    'performance_metrics': {'best_score': score}
                })
        except Exception as e:
            print(f"Error storing best config: {e}")
    
    def get_baseline_score(self) -> float:
        """Calculate baseline score using tenure-only model"""
        # Set baseline weights (tenure only)
        baseline_weights = {
            'tenure': 1.0,
            'equity': 0.0,
            'legal': 0.0,
            'permit': 0.0,
            'listing': 0.0,
            'maintenance': 0.0
        }
        
        self.scoring_engine.set_weights(baseline_weights)
        
        # Get training data
        X, y = self.get_training_data()
        if X.empty:
            return 0.0
        
        # Create time-aware splits
        splits = self.create_time_aware_splits(X, y, n_splits=3)
        if not splits:
            return 0.0
        
        # Calculate baseline score
        cv_scores = []
        for train_idx, test_idx in splits:
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            
            try:
                test_scores_df = self.scoring_engine.calculate_total_scores(X_test)
                y_scores = test_scores_df['total_score'].values
                
                precision_at_k = self.calculate_precision_at_k(y_test.values, y_scores, k=min(100, len(y_test)))
                pr_auc = self.calculate_pr_auc(y_test.values, y_scores)
                
                combined_score = 0.7 * precision_at_k + 0.3 * pr_auc
                cv_scores.append(combined_score)
                
            except Exception as e:
                print(f"Error in baseline CV fold: {e}")
                cv_scores.append(0.0)
        
        return np.mean(cv_scores) if cv_scores else 0.0
