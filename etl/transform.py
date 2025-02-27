import pandas as pd
import yaml
import os
from datetime import datetime

def load_config():
    """Load configuration from yaml file."""
    config_path = os.path.join(os.path.dirname(__file__), '../config/db_config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def validate_data(df, config):
    """Validate data based on configuration thresholds."""
    etl_config = config['etl_config']
    
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Validate year
    df = df[
        (df['Year'] >= etl_config['min_year']) & 
        (df['Year'] <= etl_config['max_year'])
    ]
    
    # Validate prices
    df = df[
        (df['Selling_Price'] >= etl_config['min_price']) & 
        (df['Selling_Price'] <= etl_config['max_price']) &
        (df['Present_Price'] >= etl_config['min_price']) & 
        (df['Present_Price'] <= etl_config['max_price'])
    ]
    
    return df

def clean_data(df):
    """Clean and transform the car prediction data."""
    if df is None:
        print("No data to transform: DataFrame is None")
        return None
    
    try:
        # Load configuration
        config = load_config()
        
        # Create a copy of the dataframe
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Convert numeric fields
        numeric_columns = {
            'Year': 'integer',
            'Selling_Price': 'float',
            'Present_Price': 'float',
            'Kms_Driven': 'integer',
            'Owner': 'integer'
        }
        
        for col, dtype in numeric_columns.items():
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if dtype == 'integer':
                df[col] = df[col].fillna(0).astype(int)
        
        # Create age feature
        current_year = datetime.now().year
        df['Car_Age'] = current_year - df['Year']
        
        # Encode categorical variables
        categorical_columns = ['Fuel_Type', 'Seller_Type', 'Transmission']
        for col in categorical_columns:
            df[col] = pd.Categorical(df[col]).codes
        
        # Validate data
        df = validate_data(df, config)
        
        print(f"Data transformed successfully. Shape after transformation: {df.shape}")
        
        # Display summary statistics
        print("\nNumeric columns summary:")
        print(df.describe())
        
        print("\nCategorical columns summary:")
        for col in categorical_columns:
            print(f"\n{col} value counts:")
            print(df[col].value_counts())
        
        return df
    except Exception as e:
        print(f"Error transforming data: {str(e)}")
        return None

if __name__ == "__main__":
    try:
        # Test the transformation
        from extract import extract_data
        
        print("Testing data transformation...")
        raw_data = extract_data()
        if raw_data is not None:
            transformed_data = clean_data(raw_data)
            if transformed_data is not None:
                print("Transformation test completed successfully")
        else:
            print("No data to test transformation")
    except Exception as e:
        print(f"Test transformation failed: {str(e)}") 