import pandas as pd
import psycopg2
import os
import sys
from datetime import datetime

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from etl.extract import load_config

def create_database():
    """Create database if it doesn't exist."""
    config = load_config()
    db_config = config['database']
    
    try:
        # Connect to default postgres database first
        conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            database='postgres',  # Connect to default database
            user=db_config['user'],
            password=db_config['password']
        )
        conn.autocommit = True  # Required for creating database
        
        with conn.cursor() as cur:
            # Check if database exists
            cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_config['name']}'")
            exists = cur.fetchone()
            
            if not exists:
                print(f"Creating database {db_config['name']}...")
                cur.execute(f"CREATE DATABASE {db_config['name']}")
                print(f"Database {db_config['name']} created successfully")
            else:
                print(f"Database {db_config['name']} already exists")
                
    except Exception as e:
        print(f"Error creating database: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()
    return True

def create_connection():
    """Create database connection."""
    config = load_config()
    db_config = config['database']
    
    try:
        # First ensure database exists
        if not create_database():
            return None
            
        conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['name'],
            user=db_config['user'],
            password=db_config['password']
        )
        print("Successfully connected to the database")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return None

def load_to_database(df):
    """Load transformed car prediction data to database."""
    if df is None:
        print("No data to load: DataFrame is None")
        return False
    
    print(f"Starting to load {len(df)} records to database")
    conn = create_connection()
    if conn is None:
        return False
    
    try:
        # Create table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS car_predictions (
            id SERIAL PRIMARY KEY,
            car_name VARCHAR(100),
            year INTEGER,
            selling_price DECIMAL(10,2),
            present_price DECIMAL(10,2),
            kms_driven INTEGER,
            fuel_type INTEGER,
            seller_type INTEGER,
            transmission INTEGER,
            owner INTEGER,
            car_age INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        with conn.cursor() as cur:
            # Create table
            cur.execute(create_table_query)
            print("Table structure verified/created")
            
            # Insert data in batches
            batch_size = 1000
            records_loaded = 0
            
            for idx in range(0, len(df), batch_size):
                batch = df.iloc[idx:idx + batch_size]
                
                for _, row in batch.iterrows():
                    try:
                        cur.execute(
                            """
                            INSERT INTO car_predictions (
                                car_name, year, selling_price, present_price, 
                                kms_driven, fuel_type, seller_type, transmission, 
                                owner, car_age
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                row['Car_Name'], row['Year'], row['Selling_Price'],
                                row['Present_Price'], row['Kms_Driven'], row['Fuel_Type'],
                                row['Seller_Type'], row['Transmission'], row['Owner'],
                                row['Car_Age']
                            )
                        )
                        records_loaded += 1
                    except Exception as e:
                        print(f"Error loading record {records_loaded + 1}: {str(e)}")
                        print(f"Problematic row: {row}")
                        continue
                
                conn.commit()
                print(f"Loaded {records_loaded} records so far...")
        
        print(f"Data loading completed. Total records loaded: {records_loaded}")
        return True
    except Exception as e:
        print(f"Error during database operations: {str(e)}")
        return False
    finally:
        conn.close()
        print("Database connection closed")

if __name__ == "__main__":
    # Test loading with sample data
    try:
        from extract import extract_data
        from transform import clean_data
        
        print("Testing the complete loading process...")
        raw_data = extract_data()
        if raw_data is not None:
            transformed_data = clean_data(raw_data)
            load_to_database(transformed_data)
        else:
            print("No data to test loading")
    except Exception as e:
        print(f"Test loading failed: {str(e)}") 