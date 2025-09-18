"""
Backtesting Module for Augur Seller Scoring
Implements precision@K, PR-AUC, and calibration analysis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    precision_score, recall_score, average_precision_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.calibration import CalibratedClassifierCV

# Try to import calibration_curve, fallback if not available
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    # Fallback implementation for older sklearn versions
    def calibration_curve(y_true, y_prob, n_bins=5):
        """Simple calibration curve implementation"""
        import numpy as np
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return np.linspace(0, 1, n_bins), np.linspace(0, 1, n_bins)
import warnings
warnings.filterwarnings('ignore')

from scoring.engine import ScoringEngine

class Backtester:
    def __init__(self, db_engine):
        """Initialize the backtester"""
        self.db_engine = db_engine
        self.scoring_engine = ScoringEngine(db_engine)
        
    def get_backtest_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get data for backtesting within date range"""
        query = """
        SELECT p.*, 
               CASE WHEN p.status = 'Sold' THEN 1 ELSE 0 END as sold_label,
               p.report_date as outcome_date
        FROM properties p
        WHERE p.report_date BETWEEN %(start_date)s AND %(end_date)s
        AND p.status IN ('Sold', 'Off Market', 'Active', 'Pending', 'None')
        ORDER BY p.report_date
        """
        
        try:
            with self.db_engine.connect() as conn:
                df = pd.read_sql(query, conn, params={'start_date': start_date, 'end_date': end_date})
        except Exception as e:
            print(f"Error getting backtest data: {e}")
            return pd.DataFrame()
        
        return df
    
    def calculate_precision_at_k(self, y_true: np.ndarray, y_scores: np.ndarray, k_values: List[int] = None) -> Dict[int, float]:
        """Calculate precision@K for multiple K values"""
        if k_values is None:
            k_values = [10, 25, 50, 100, 200]
        
        precision_at_k = {}
        
        for k in k_values:
            if k > len(y_true):
                k = len(y_true)
            
            if k == 0:
                precision_at_k[k] = 0.0
                continue
            
            # Get top K predictions
            top_k_indices = np.argsort(y_scores)[-k:]
            top_k_labels = y_true[top_k_indices]
            
            # Calculate precision
            if len(top_k_labels) == 0:
                precision_at_k[k] = 0.0
            else:
                precision_at_k[k] = np.sum(top_k_labels) / len(top_k_labels)
        
        return precision_at_k
    
    def calculate_recall_at_k(self, y_true: np.ndarray, y_scores: np.ndarray, k_values: List[int] = None) -> Dict[int, float]:
        """Calculate recall@K for multiple K values"""
        if k_values is None:
            k_values = [10, 25, 50, 100, 200]
        
        recall_at_k = {}
        total_positive = np.sum(y_true)
        
        if total_positive == 0:
            return {k: 0.0 for k in k_values}
        
        for k in k_values:
            if k > len(y_true):
                k = len(y_true)
            
            if k == 0:
                recall_at_k[k] = 0.0
                continue
            
            # Get top K predictions
            top_k_indices = np.argsort(y_scores)[-k:]
            top_k_labels = y_true[top_k_indices]
            
            # Calculate recall
            if len(top_k_labels) == 0:
                recall_at_k[k] = 0.0
            else:
                recall_at_k[k] = np.sum(top_k_labels) / total_positive
        
        return recall_at_k
    
    def calculate_pr_auc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Calculate Precision-Recall AUC"""
        if len(np.unique(y_true)) < 2:
            return 0.0
        
        try:
            pr_auc = average_precision_score(y_true, y_scores)
            return pr_auc
        except:
            return 0.0
    
    def calculate_roc_auc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Calculate ROC AUC"""
        if len(np.unique(y_true)) < 2:
            return 0.0
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            return roc_auc
        except:
            return 0.0
    
    def calculate_calibration_metrics(self, y_true: np.ndarray, y_scores: np.ndarray, n_bins: int = 10) -> Dict:
        """Calculate calibration metrics"""
        try:
            # Convert scores to probabilities (normalize to 0-1)
            y_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-8)
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_proba, n_bins=n_bins
            )
            
            # Calculate Brier score
            brier_score = np.mean((y_proba - y_true) ** 2)
            
            # Calculate reliability (calibration error)
            reliability = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            
            return {
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value,
                'brier_score': brier_score,
                'reliability': reliability
            }
        except Exception as e:
            print(f"Error calculating calibration metrics: {e}")
            return {
                'fraction_of_positives': np.array([]),
                'mean_predicted_value': np.array([]),
                'brier_score': 1.0,
                'reliability': 1.0
            }
    
    def run_backtest(self, start_date: datetime, end_date: datetime, 
                    weights: Dict = None, time_decay_half_life: int = 90) -> Dict:
        """Run complete backtest analysis"""
        
        # Set scoring parameters
        if weights:
            self.scoring_engine.set_weights(weights)
        self.scoring_engine.set_time_decay(time_decay_half_life)
        
        # Get backtest data
        df = self.get_backtest_data(start_date, end_date)
        if df.empty:
            return {'error': 'No data available for backtest period'}
        
        # Calculate scores
        try:
            scores_df = self.scoring_engine.calculate_total_scores(df)
            y_true = df['sold_label'].values
            y_scores = scores_df['total_score'].values
        except Exception as e:
            return {'error': f'Error calculating scores: {e}'}
        
        # Calculate metrics
        precision_at_k = self.calculate_precision_at_k(y_true, y_scores)
        recall_at_k = self.calculate_recall_at_k(y_true, y_scores)
        pr_auc = self.calculate_pr_auc(y_true, y_scores)
        roc_auc = self.calculate_roc_auc(y_true, y_scores)
        calibration_metrics = self.calculate_calibration_metrics(y_true, y_scores)
        
        # Calculate baseline (tenure-only) for comparison
        baseline_weights = {
            'tenure': 1.0, 'equity': 0.0, 'legal': 0.0,
            'permit': 0.0, 'listing': 0.0, 'maintenance': 0.0
        }
        self.scoring_engine.set_weights(baseline_weights)
        baseline_scores_df = self.scoring_engine.calculate_total_scores(df)
        baseline_y_scores = baseline_scores_df['total_score'].values
        baseline_precision_at_k = self.calculate_precision_at_k(y_true, baseline_y_scores)
        baseline_pr_auc = self.calculate_pr_auc(y_true, baseline_y_scores)
        
        # Restore original weights
        if weights:
            self.scoring_engine.set_weights(weights)
        
        return {
            'precision_at_k': precision_at_k,
            'recall_at_k': recall_at_k,
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'calibration_metrics': calibration_metrics,
            'baseline_precision_at_k': baseline_precision_at_k,
            'baseline_pr_auc': baseline_pr_auc,
            'total_properties': len(df),
            'positive_properties': np.sum(y_true),
            'negative_properties': len(df) - np.sum(y_true),
            'start_date': start_date,
            'end_date': end_date,
            'weights': weights,
            'time_decay_half_life': time_decay_half_life
        }
    
    def create_precision_recall_plot(self, backtest_results: Dict) -> go.Figure:
        """Create precision-recall curve plot"""
        # Get data
        df = self.get_backtest_data(backtest_results['start_date'], backtest_results['end_date'])
        if df.empty:
            return go.Figure()
        
        # Calculate scores
        scores_df = self.scoring_engine.calculate_total_scores(df)
        y_true = df['sold_label'].values
        y_scores = scores_df['total_score'].values
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        # Create plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name='Precision-Recall Curve',
            line=dict(color='blue', width=2)
        ))
        
        # Add baseline (random classifier)
        baseline_precision = np.sum(y_true) / len(y_true)
        fig.add_hline(
            y=baseline_precision,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Baseline (Random): {baseline_precision:.3f}"
        )
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def create_calibration_plot(self, backtest_results: Dict) -> go.Figure:
        """Create calibration plot"""
        calibration_metrics = backtest_results['calibration_metrics']
        
        if len(calibration_metrics['fraction_of_positives']) == 0:
            return go.Figure()
        
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='red', dash='dash')
        ))
        
        # Actual calibration curve
        fig.add_trace(go.Scatter(
            x=calibration_metrics['mean_predicted_value'],
            y=calibration_metrics['fraction_of_positives'],
            mode='lines+markers',
            name='Model Calibration',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Calibration Plot',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def create_precision_at_k_plot(self, backtest_results: Dict) -> go.Figure:
        """Create precision@K comparison plot"""
        precision_at_k = backtest_results['precision_at_k']
        baseline_precision_at_k = backtest_results['baseline_precision_at_k']
        
        k_values = list(precision_at_k.keys())
        model_precisions = list(precision_at_k.values())
        baseline_precisions = [baseline_precision_at_k.get(k, 0) for k in k_values]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=k_values,
            y=model_precisions,
            mode='lines+markers',
            name='Optimized Model',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=k_values,
            y=baseline_precisions,
            mode='lines+markers',
            name='Baseline (Tenure Only)',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Precision@K Comparison',
            xaxis_title='K (Number of Top Predictions)',
            yaxis_title='Precision',
            xaxis=dict(type='log'),
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def save_backtest_results(self, backtest_results: Dict, model_config_id: int = None):
        """Save backtest results to database"""
        insert_query = """
        INSERT INTO backtest_results 
        (model_config_id, backtest_date, precision_at_k, recall_at_k, pr_auc, 
         calibration_score, top_k_count, total_properties, time_period_start, time_period_end)
        VALUES 
        (:model_config_id, :backtest_date, :precision_at_k, :recall_at_k, :pr_auc,
         :calibration_score, :top_k_count, :total_properties, :time_period_start, :time_period_end)
        """
        
        # Calculate summary metrics
        precision_at_k_100 = backtest_results['precision_at_k'].get(100, 0.0)
        recall_at_k_100 = backtest_results['recall_at_k'].get(100, 0.0)
        pr_auc = backtest_results['pr_auc']
        calibration_score = 1.0 - backtest_results['calibration_metrics']['reliability']  # Higher is better
        
        try:
            with self.db_engine.begin() as conn:
                conn.execute(insert_query, {
                    'model_config_id': model_config_id,
                    'backtest_date': datetime.now().date(),
                    'precision_at_k': precision_at_k_100,
                    'recall_at_k': recall_at_k_100,
                    'pr_auc': pr_auc,
                    'calibration_score': calibration_score,
                    'top_k_count': 100,
                    'total_properties': backtest_results['total_properties'],
                    'time_period_start': backtest_results['start_date'].date(),
                    'time_period_end': backtest_results['end_date'].date()
                })
        except Exception as e:
            print(f"Error saving backtest results: {e}")
    
    def get_top_k_properties(self, k: int = 100, weights: Dict = None, 
                           time_decay_half_life: int = 90) -> pd.DataFrame:
        """Get top K scoring properties for export"""
        
        # Set scoring parameters
        if weights:
            self.scoring_engine.set_weights(weights)
        self.scoring_engine.set_time_decay(time_decay_half_life)
        
        # Get all properties
        query = """
        SELECT * FROM properties 
        WHERE status IN ('Off Market', 'Active', 'Pending', 'None')
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
            scores_df = self.scoring_engine.calculate_total_scores(df)
            # Get top K
            top_k_df = scores_df.nlargest(k, 'total_score')
            return top_k_df
        except Exception as e:
            print(f"Error calculating scores: {e}")
            return pd.DataFrame()
