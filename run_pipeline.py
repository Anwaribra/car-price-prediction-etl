from etl.extract import extract_data
from etl.transform import clean_data
from etl.load import load_to_database
from prediction.model import CarPricePredictor

def run_pipeline(train_model=True):
    """Execute the complete ETL pipeline with optional prediction."""
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
        print("\nTraining prediction model...")
        predictor = CarPricePredictor()
        predictor.train(transformed_data)
        
        print("\nMaking predictions...")
        predictions = predictor.predict(transformed_data)
        if predictions is not None:
            print("\nSample predictions:")
            print(predictions[['Car_Name', 'Selling_Price', 'Predicted_Price', 
                            'Price_Lower_Bound', 'Price_Upper_Bound']].head())
            
            # Optionally save predictions to database
            load_to_database(predictions, table_name='car_predictions_results')
    
    print("Pipeline completed successfully!")
    return True

if __name__ == "__main__":
    run_pipeline() 