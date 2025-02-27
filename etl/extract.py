import pandas as pd
import yaml
import os

def load_config():
    """Load configuration from yaml file."""
    config_path = os.path.join(os.path.dirname(__file__), '../config/db_config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def extract_data():
    """Extract data from the car prediction CSV file."""
    try:
        file_path = os.path.join(os.path.dirname(__file__), 
                                '../data/car_prediction_data.csv')
        
        df = pd.read_csv(file_path)
        print(f"Data extracted successfully. Shape: {df.shape}")
        
        # Display first few rows and data info
        print("\nFirst few rows of the data:")
        print(df.head())
        print("\nDataset Info:")
        print(df.info())
        
        return df
    except Exception as e:
        print(f"Error extracting data: {str(e)}")
        return None

if __name__ == "__main__":
    extract_data() 