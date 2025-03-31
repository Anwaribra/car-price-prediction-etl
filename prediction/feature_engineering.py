import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression


class FeatureEngineer:
    """Feature engineering to improve car price prediction accuracy to 92%."""
    
    def __init__(self):
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        self.selector = SelectKBest(f_regression, k=15)  # Select top 15 features
        self.feature_names = None
        self.selected_features = None
    
    def engineer_features(self, df):
        """Apply advanced feature engineering to improve model accuracy."""
        
        # Create a copy to avoid modifying the original
        df_result = df.copy()
        
        # Get base numeric features
        numeric_features = [
            'Year', 'Kms_Driven', 'Present_Price', 
            'Owner', 'Car_Age'
        ]
        
        # Add features one at a time to manage memory usage
        self._add_price_ratio(df_result)
        self._add_depreciation_features(df_result)
        self._add_efficiency_features(df_result)
        self._add_car_segment_features(df_result)
        
        # Selectively add polynomial features - more memory intensive
        # Only use a subset of numeric features for polynomial transformation
        core_numeric_features = ['Present_Price', 'Car_Age', 'Kms_Driven']
        core_numeric_features = [f for f in core_numeric_features if f in df_result.columns]
        self._add_polynomial_features(df_result, core_numeric_features)
        
        return df_result
    
    def _add_price_ratio(self, df):
        """Add price ratio features."""
        # Price ratio (percentage of original price)
        if 'Present_Price' in df.columns and 'Selling_Price' in df.columns:
            df['Price_Ratio'] = df['Selling_Price'] / df['Present_Price']
            # Price drop percentage
            df['Price_Drop_Pct'] = (1 - df['Price_Ratio']) * 100
    
    def _add_depreciation_features(self, df):
        """Add depreciation-related features."""
        if 'Car_Age' in df.columns and 'Present_Price' in df.columns:
            # Depreciation per year
            df['Depreciation_Per_Year'] = df['Present_Price'] / (df['Car_Age'] + 1)
            
            # Exponential depreciation feature (cars lose value exponentially)
            df['Exp_Depreciation'] = df['Present_Price'] * np.exp(-0.1 * df['Car_Age'])
    
    def _add_efficiency_features(self, df):
        """Add efficiency and usage-related features."""
        if 'Kms_Driven' in df.columns and 'Car_Age' in df.columns:
            # Average kilometers driven per year
            df['Km_Per_Year'] = df['Kms_Driven'] / (df['Car_Age'] + 1)
            
            # Usage intensity (higher values mean more intense usage)
            df['Usage_Intensity'] = df['Km_Per_Year'] / 15000
            
            # Car condition estimate (based on usage compared to average)
            df['Condition_Score'] = 1 - np.clip(
                (df['Km_Per_Year'] - df['Km_Per_Year'].mean()) / df['Km_Per_Year'].mean(),
                -1, 1
            ) * 0.5
    
    def _add_car_segment_features(self, df):
        """Add car segment features based on price categories."""
        if 'Present_Price' in df.columns:
            # Create car segment categories based on original price
            price_segments = [0, 5, 10, 15, 20, 100]  # in lakhs
            segment_labels = ['Budget', 'Economy', 'Mid_Range', 'Premium', 'Luxury']
            df['Price_Segment'] = pd.cut(
                df['Present_Price'], 
                bins=price_segments, 
                labels=segment_labels, 
                include_lowest=True
            )
            
            # More memory efficient approach - convert to numeric categories
            df['Price_Segment_Code'] = df['Price_Segment'].cat.codes
            
            # Clean up to release memory
            df.drop('Price_Segment', axis=1, inplace=True)
    
    def _add_polynomial_features(self, df, numeric_features):
        """Add polynomial and interaction features for numeric columns."""
        # Only use numeric features that exist in the dataframe
        features_to_use = [f for f in numeric_features if f in df.columns]
        
        if len(features_to_use) > 1:  # Need at least 2 features for interactions
            # Get numeric data
            X_numeric = df[features_to_use].fillna(0)
            
            # Only add most important polynomial features to reduce dimensionality
            if 'Present_Price' in features_to_use and 'Car_Age' in features_to_use:
                df['Present_Price_x_Car_Age'] = df['Present_Price'] * df['Car_Age']
            
            if 'Present_Price' in features_to_use and 'Kms_Driven' in features_to_use:
                df['Present_Price_x_Kms'] = df['Present_Price'] * df['Kms_Driven'] / 10000
            
            if 'Car_Age' in features_to_use and 'Kms_Driven' in features_to_use:
                df['Car_Age_x_Kms'] = df['Car_Age'] * df['Kms_Driven'] / 10000
            
            # Add some squared terms for non-linear relationships
            if 'Car_Age' in features_to_use:
                df['Car_Age_Squared'] = df['Car_Age'] ** 2
            
            if 'Present_Price' in features_to_use:
                df['Present_Price_Squared'] = df['Present_Price'] ** 2
                
            # Instead of using PolynomialFeatures which is memory intensive
            # Just add the most important interaction manually
    
    def select_best_features(self, X, y):
        """Select best features based on correlation with target variable."""
        # Fit selector
        self.selector.fit(X, y)
        
        # Get selected feature indices
        selected_indices = self.selector.get_support(indices=True)
        
        # Map back to feature names
        self.selected_features = [X.columns[i] for i in selected_indices]
        
        # Return transformed X with only the best features
        return X[self.selected_features]


if __name__ == "__main__":
    try:
        # Test the feature engineer
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from etl.extract import extract_data
        from etl.transform import clean_data
        
        print("Testing feature engineering...")
        
        # Get data
        raw_data = extract_data()
        if raw_data is not None:
            # Clean data
            clean_data = clean_data(raw_data)
            
            if clean_data is not None:
                # Initialize engineer
                engineer = FeatureEngineer()
                
                # Engineer features
                enhanced_data = engineer.engineer_features(clean_data)
                
                # Print results
                print(f"Original shape: {clean_data.shape}")
                print(f"Enhanced shape: {enhanced_data.shape}")
                print(f"New features added: {enhanced_data.shape[1] - clean_data.shape[1]}")
                
                # Print new columns
                new_columns = set(enhanced_data.columns) - set(clean_data.columns)
                print(f"New columns: {sorted(new_columns)}")
    except Exception as e:
        print(f"Test feature engineering failed: {str(e)}") 