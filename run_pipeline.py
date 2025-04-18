from etl.extract import extract_data
from etl.transform import clean_data
from etl.load import load_to_database
from prediction.model import CarPricePredictor
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
import gc  # For garbage collection

def run_pipeline(train_model=True, validate_accuracy=True):
    """Execute the complete ETL pipeline with advanced prediction and validation."""
    print("Starting ETL pipeline...")
    
    # Extract
    raw_data = extract_data()
    if raw_data is None:
        print("Extraction failed. Stopping pipeline.")
        return False
    
    # Transform
    transformed_data = clean_data(raw_data)
    if transformed_data is None:
        print("Transformation failed. Stopping pipeline.")
        return False
    
    # Load
    success = load_to_database(transformed_data)
    if not success:
        print("Loading failed. Pipeline completed with errors.")
        return False
    
    # Prediction
    if train_model:
        try:
            print("\nTraining prediction model...")
            predictor = CarPricePredictor()
            
            # Train with memory efficient approach
            target_achieved = predictor.train(transformed_data)
            
            # Force garbage collection after training
            gc.collect()
            
            if target_achieved:
                print("\nTarget accuracy of 92% achieved during training!")
            else:
                print("\nStill working to reach 92% accuracy target...")
            
            print("\nMaking predictions...")
            predictions = predictor.predict(transformed_data)
            
            if predictions is not None:
                print("\nSample predictions:")
                print(predictions[['Car_Name', 'Selling_Price', 'Predicted_Price', 
                                'Price_Lower_Bound', 'Price_Upper_Bound']].head())
                
                # Evaluate model to check 92% accuracy target
                if validate_accuracy and 'Selling_Price' in predictions.columns:
                    r2 = r2_score(predictions['Selling_Price'], predictions['Predicted_Price'])
                    print(f"\nFinal Model Accuracy (R² Score): {r2:.4f}")
                    
                    if r2 >= 0.92:
                        print("✓ TARGET ACHIEVED: Model accuracy is at or above 92%")
                        
                        # Create evaluation plots directory
                        plots_dir = os.path.join('prediction', 'tracking', 'plots')
                        os.makedirs(plots_dir, exist_ok=True)
                        
                        try:
                            # Create actual vs predicted plot
                            plt.figure(figsize=(10, 6))
                            plt.scatter(predictions['Selling_Price'], predictions['Predicted_Price'], alpha=0.5)
                            plt.plot([predictions['Selling_Price'].min(), predictions['Selling_Price'].max()], 
                                    [predictions['Selling_Price'].min(), predictions['Selling_Price'].max()], 
                                    'r--')
                            plt.xlabel('Actual Price')
                            plt.ylabel('Predicted Price')
                            plt.title(f'Actual vs Predicted Prices (R² = {r2:.4f})')
                            plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted.png'))
                            plt.close()
                            
                            # Create error distribution plot
                            errors = predictions['Selling_Price'] - predictions['Predicted_Price']
                            plt.figure(figsize=(10, 6))
                            plt.hist(errors, bins=30, alpha=0.7)
                            plt.axvline(0, color='r', linestyle='--')
                            plt.xlabel('Prediction Error')
                            plt.ylabel('Frequency')
                            plt.title('Distribution of Prediction Errors')
                            plt.savefig(os.path.join(plots_dir, 'error_distribution.png'))
                            plt.close()
                            
                            print(f"Performance visualization saved to {plots_dir}")
                        except Exception as e:
                            print(f"Error creating visualizations: {e}")
                    else:
                        print(f"✗ TARGET NOT MET: Final model accuracy is {r2:.4f}, which is below the 92% target")
                
                # Optionally save predictions to database
                try:
                    load_to_database(predictions, table_name='car_predictions_results')
                except Exception as e:
                    print(f"Error saving predictions to database: {e}")
        except Exception as e:
            print(f"Error in prediction process: {e}")
            return False
    
    print("Pipeline completed successfully!")
    return True

if __name__ == "__main__":
    run_pipeline() 