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
        self._setup_tracking_dir()

    def _setup_tracking_dir(self):
        """Create tracking directory if it doesn't exist."""
        os.makedirs(self.tracking_dir, exist_ok=True)

    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive model metrics."""
        metrics = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'mape': float(mean_absolute_percentage_error(y_true, y_pred)),
                'r2': float(r2_score(y_true, y_pred)),
                'sample_size': len(y_true)
            }
        }
        return metrics

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
            
            print(f"Metrics logged successfully for model version: {model_version}")
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

            # Generate summary
            summary = {
                'overall_metrics': metrics_history,
                'predictions_summary': {
                    'total_predictions': len(predictions_df),
                    'mean_error': float(predictions_df['error'].mean()),
                    'median_error': float(predictions_df['error'].median()),
                    'error_std': float(predictions_df['error'].std()),
                    'mean_error_percentage': float(predictions_df['error_percentage'].mean())
                },
                'top_underestimated': predictions_df.nlargest(5, 'error')[
                    ['car_name', 'actual_price', 'predicted_price', 'error_percentage']
                ].to_dict('records'),
                'top_overestimated': predictions_df.nsmallest(5, 'error')[
                    ['car_name', 'actual_price', 'predicted_price', 'error_percentage']
                ].to_dict('records')
            }

            return summary
        except Exception as e:
            print(f"Error generating performance summary: {str(e)}")
            return None 