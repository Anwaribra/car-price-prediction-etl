import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

class ModelTracker:
    def __init__(self):
        self.tracking_dir = os.path.join(os.path.dirname(__file__), 'tracking')
        self.metrics_file = os.path.join(self.tracking_dir, 'model_metrics.json')
        self.performance_file = os.path.join(self.tracking_dir, 'performance_history.csv')
        self.accuracy_log_file = os.path.join(self.tracking_dir, 'accuracy_log.csv')
        self.target_accuracy = 0.92  # 92% accuracy target
        self._setup_tracking_dir()

    def _setup_tracking_dir(self):
        """Create tracking directory if it doesn't exist."""
        os.makedirs(self.tracking_dir, exist_ok=True)

    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive model metrics."""
        r2 = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = float(mean_absolute_percentage_error(y_true, y_pred))
        
        # Log whether we met the accuracy target
        target_met = r2 >= self.target_accuracy
        self._log_accuracy_check(r2, target_met)
        
        metrics = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'rmse': rmse,
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'mape': mape,
                'r2': r2,
                'r2_percent': r2 * 100,  # R² as percentage for easier comparison with target
                'target_accuracy_met': target_met,
                'sample_size': len(y_true)
            }
        }
        return metrics

    def _log_accuracy_check(self, r2_score, target_met):
        """Log accuracy check results to track progress toward 92% target."""
        accuracy_log = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'r2_score': r2_score,
            'r2_percent': r2_score * 100,
            'target_met': target_met,
            'gap_to_target': max(0, self.target_accuracy - r2_score)
        }
        
        # Convert to DataFrame for single row
        log_df = pd.DataFrame([accuracy_log])
        
        # Append to CSV
        if os.path.exists(self.accuracy_log_file):
            log_df.to_csv(self.accuracy_log_file, mode='a', header=False, index=False)
        else:
            log_df.to_csv(self.accuracy_log_file, index=False)

    def log_metrics(self, metrics, model_version='latest'):
        """Log model metrics to JSON file."""
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    history = json.load(f)
            else:
                history = {}

            history[model_version] = metrics
            
            with open(self.metrics_file, 'w') as f:
                json.dump(history, f, indent=4)
            
            r2_percent = metrics['metrics']['r2'] * 100
            target_status = "✓ MET" if metrics['metrics']['r2'] >= self.target_accuracy else "✗ NOT MET"
            print(f"Metrics logged successfully for model version: {model_version}")
            print(f"Accuracy: {r2_percent:.2f}% - Target (92%): {target_status}")
        except Exception as e:
            print(f"Error logging metrics: {str(e)}")

    def log_predictions(self, y_true, y_pred, car_names):
        """Log individual prediction performance."""
        try:
            df = pd.DataFrame({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'car_name': car_names,
                'actual_price': y_true,
                'predicted_price': y_pred,
                'error': y_true - y_pred,
                'absolute_error': np.abs(y_true - y_pred),
                'squared_error': (y_true - y_pred) ** 2,
                'error_percentage': ((y_true - y_pred) / y_true) * 100
            })

            if os.path.exists(self.performance_file):
                df.to_csv(self.performance_file, mode='a', header=False, index=False)
            else:
                df.to_csv(self.performance_file, index=False)

            print("Prediction performance logged successfully")
        except Exception as e:
            print(f"Error logging predictions: {str(e)}")

    def get_performance_summary(self):
        """Generate performance summary report."""
        try:
            # Read metrics history
            with open(self.metrics_file, 'r') as f:
                metrics_history = json.load(f)

            # Read prediction history
            predictions_df = pd.read_csv(self.performance_file)
            
            # Calculate r2 directly from predictions
            if 'actual_price' in predictions_df.columns and 'predicted_price' in predictions_df.columns:
                overall_r2 = r2_score(predictions_df['actual_price'], predictions_df['predicted_price'])
                target_status = "✓ MET" if overall_r2 >= self.target_accuracy else "✗ NOT MET"
            else:
                overall_r2 = None
                target_status = "Unknown"

            # Generate summary
            summary = {
                'overall_metrics': metrics_history,
                'predictions_summary': {
                    'total_predictions': len(predictions_df),
                    'mean_error': float(predictions_df['error'].mean()),
                    'median_error': float(predictions_df['error'].median()),
                    'error_std': float(predictions_df['error'].std()),
                    'mean_error_percentage': float(predictions_df['error_percentage'].mean()),
                    'r2_score': overall_r2,
                    'r2_percent': overall_r2 * 100 if overall_r2 is not None else None,
                    'target_accuracy_met': target_status
                },
                'top_underestimated': predictions_df.nlargest(5, 'error')[
                    ['car_name', 'actual_price', 'predicted_price', 'error_percentage']
                ].to_dict('records'),
                'top_overestimated': predictions_df.nsmallest(5, 'error')[
                    ['car_name', 'actual_price', 'predicted_price', 'error_percentage']
                ].to_dict('records'),
                'accuracy_vs_target': {
                    'target': self.target_accuracy * 100,
                    'current': overall_r2 * 100 if overall_r2 is not None else None,
                    'status': target_status
                }
            }

            return summary
        except Exception as e:
            print(f"Error generating performance summary: {str(e)}")
            return None 