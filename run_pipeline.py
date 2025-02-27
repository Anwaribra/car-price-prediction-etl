from etl.extract import extract_data
from etl.transform import clean_data
from etl.load import load_to_database

def run_pipeline():
    """Execute the complete ETL pipeline."""
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
    
    print("Pipeline completed successfully!")
    return True

if __name__ == "__main__":
    run_pipeline() 