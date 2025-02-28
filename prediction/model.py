import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction.model_tracking import ModelTracker

class CarPricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.model_path = os.path.join(os.path.dirname(__file__), 'models/rf_model.joblib')
        self.scaler_path = os.path.join(os.path.dirname(__file__), 'models/scaler.joblib')
        self.tracker = ModelTracker()

    def prepare_features(self, df):
        """Prepare features for model training/prediction."""
        
        features = [
            'Year', 'Kms_Driven', 'Present_Price', 
            'Fuel_Type', 'Seller_Type', 'Transmission', 
            'Owner', 'Car_Age'
        ]
        
        return df[features]

    def train(self, df):
        """Train the model with provided data."""
        
        X = self.prepare_features(df)
        y = df['Selling_Price']

        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Make predictions on test set
        y_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Model Performance Metrics:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R2 Score: {r2:.2f}")

        # Calculate and log metrics
        metrics = self.tracker.calculate_metrics(y_test, y_pred)
        self.tracker.log_metrics(metrics)
        
        # Log individual predictions
        self.tracker.log_predictions(
            y_test, 
            y_pred,
            df.loc[y_test.index, 'Car_Name']
        )

        # Print performance summary
        print("\nModel Performance Summary:")
        summary = self.tracker.get_performance_summary()
        if summary:
            print("\nOverall Metrics:")
            print(f"R2 Score: {summary['overall_metrics']['latest']['metrics']['r2']:.3f}")
            print(f"RMSE: {summary['overall_metrics']['latest']['metrics']['rmse']:.2f}")
            print(f"MAPE: {summary['overall_metrics']['latest']['metrics']['mape']:.2%}")
            
            print("\nPrediction Analysis:")
            print(f"Mean Error Percentage: {summary['predictions_summary']['mean_error_percentage']:.2f}%")
            
            print("\nTop 5 Underestimated Cars:")
            for car in summary['top_underestimated']:
                print(f"{car['car_name']}: Actual={car['actual_price']:.2f}, "
                      f"Predicted={car['predicted_price']:.2f}, "
                      f"Error={car['error_percentage']:.2f}%")

        # Save model and scaler
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"Model saved to {self.model_path}")

    def predict(self, df):
        """Make predictions using trained model."""
        try:
            # Load model and scaler if not already loaded
            if not hasattr(self.model, 'predict'):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)

            # Prepare features
            X = self.prepare_features(df)
            X_scaled = self.scaler.transform(X)

            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            # Add predictions to dataframe
            df['Predicted_Price'] = predictions
            
            # Calculate prediction intervals (simple approach)
            feature_importance = self.model.feature_importances_
            uncertainty = np.std(predictions) * (1 - np.mean(feature_importance))
            df['Price_Lower_Bound'] = predictions - 2 * uncertainty
            df['Price_Upper_Bound'] = predictions + 2 * uncertainty

            # Log predictions if actual prices are available
            if 'Selling_Price' in df.columns:
                self.tracker.log_predictions(
                    df['Selling_Price'],
                    predictions,
                    df['Car_Name']
                )

            return df
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            return None

if __name__ == "__main__":
    # Test the predictor
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from etl.extract import extract_data
        from etl.transform import clean_data

        print("Testing the prediction model...")
        
        # Get and prepare data
        raw_data = extract_data()
        if raw_data is not None:
            clean_data = clean_data(raw_data)
            if clean_data is not None:
                # Initialize and train model
                predictor = CarPricePredictor()
                predictor.train(clean_data)
                
                # Make predictions
                predictions = predictor.predict(clean_data)
                if predictions is not None:
                    print("\nSample predictions:")
                    print(predictions[['Car_Name', 'Selling_Price', 'Predicted_Price', 
                                    'Price_Lower_Bound', 'Price_Upper_Bound']].head())
    except Exception as e:
        print(f"Test prediction failed: {str(e)}") 