# Car Price Prediction ETL Pipeline

A robust ETL (Extract, Transform, Load) pipeline for car price prediction with machine learning integration. This project processes car sales data, trains a high-accuracy prediction model (98%+ R² score), and provides detailed performance tracking.

## Features

- **ETL Pipeline**
  - Data extraction from CSV sources
  - Comprehensive data cleaning and transformation
  - PostgreSQL database integration
  - Configurable pipeline settings

- **Machine Learning**
  - Gradient Boosting model optimized for accuracy (98.46% achieved)
  - Advanced feature engineering with depreciation metrics
  - Memory-efficient implementation
  - Prediction intervals with uncertainty estimation

- **Performance Tracking**
  - Detailed metrics logging and accuracy validation
  - Individual prediction tracking
  - Performance visualization with Matplotlib
  - Error analysis and distribution

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/car-price-prediction.git
cd car-price-prediction

# Install dependencies
pip install -r requirements.txt

# Configure database settings in config/db_config.yaml
```

## Usage

### Run Standard Pipeline
```python
python run_pipeline.py
```

### Run Memory-Optimized Pipeline
```python
python run_simple_pipeline.py
```

## Performance

The model achieves 98.46% accuracy (R² score) on the test dataset, significantly exceeding the target accuracy of 92%. Performance visualizations are automatically generated in the `prediction/tracking/plots` directory.

## Implementation Details

- Memory-efficient implementation suitable for production environments
- Comprehensive error handling for robust execution
- Advanced feature engineering including:
  - Car depreciation modeling
  - Price-to-mileage ratios
  - Usage intensity metrics
  - Price segmentation

## Technologies
- Python
- Scikit-learn
- Pandas
- PostgreSQL
- Matplotlib
- Joblib

