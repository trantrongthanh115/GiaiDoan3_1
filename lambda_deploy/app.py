import os
import json
import joblib
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Union
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("stateless_ml_api")

# Initialize FastAPI application
app = FastAPI(
    title="Fashion Retail Demand Forecasting - Stateless ML API",
    description="Stateless ML API that runs predictions using a LightGBM model. No DB connection required.",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. LOAD MODEL & PREPROCESSORS AT STARTUP
# ==========================================
model = None
scaler = None
encoder = None

def get_model_path(filename: str, fallback: str) -> str:
    """Finds the model file in the current directory or parent directory."""
    if os.path.exists(filename):
        return filename
    if os.path.exists(fallback):
        return fallback
    # Check absolute directory where this app.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path1 = os.path.join(base_dir, filename)
    if os.path.exists(path1):
        return path1
    path2 = os.path.join(base_dir, fallback)
    if os.path.exists(path2):
        return path2
    return filename

model_file = get_model_path("lightgbm_model.pkl", "lightgbm_demand_model.pkl")
scaler_file = get_model_path("standard_scaler.pkl", "standard_scaler.pkl")
encoder_file = get_model_path("label_encoder.pkl", "label_encoders.pkl")

try:
    if os.path.exists(model_file):
        model = joblib.load(model_file)
        logger.info(f"Loaded LightGBM model from {model_file}")
    else:
        logger.error(f"Model file not found at {model_file}")

    if os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
        logger.info(f"Loaded Standard Scaler from {scaler_file}")
    else:
        logger.error(f"Scaler file not found at {scaler_file}")

    if os.path.exists(encoder_file):
        encoder = joblib.load(encoder_file)
        logger.info(f"Loaded Label Encoders from {encoder_file}")
    else:
        logger.error(f"Encoder file not found at {encoder_file}")
except Exception as e:
    logger.error(f"Error loading models or preprocessors: {e}")

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def safe_label_transform(label_encoder, value: Any) -> int:
    """Transforms categoricals safely, falling back if value is unseen."""
    classes_list = label_encoder.classes_
    
    # Normalize to string if class values are strings
    if len(classes_list) > 0 and isinstance(classes_list[0], str):
        value_str = str(value)
        if value_str in classes_list:
            return int(label_encoder.transform([value_str])[0])
        if value in classes_list:
            return int(label_encoder.transform([value])[0])
    else:
        if value in classes_list:
            return int(label_encoder.transform([value])[0])
            
    # Fallback to the first class in encoder classes if label is unseen
    fallback_val = classes_list[0]
    return int(label_encoder.transform([fallback_val])[0])

def parse_store_id(store_id_raw: Any) -> int:
    """Parses store ID strings (like STORE_001, STORE_1, or 1) to integer."""
    if isinstance(store_id_raw, str):
        clean_str = store_id_raw.upper().replace('STORE_', '')
        try:
            return int(clean_str)
        except ValueError:
            return 1
    try:
        return int(store_id_raw)
    except (ValueError, TypeError):
        return 1

# Required features in correct column order
FEATURE_COLS = [
    'store_id', 'product_id', 'sku', 'size', 'color', 
    'total_lifespan', 's_total_lifespan', 'sku_store_coverage', 'product_store_coverage', 
    's_days_active', 's_selling_days_count', 's_sales_velocity', 
    's2_days_active', 's2_selling_days_count', 's2_sales_velocity', 
    'avg_usd_price', 'total_discount_avg', 'lag_1d', 'lag_7d', 
    'rolling_mean_7d', 'rolling_std_7d', 'category', 'sub_category', 
    'color_type', 'country', 'city', 'num_distinct_products', 'num_distinct_skus', 
    'month', 'day_of_week', 'day'
]

def make_predictions(records: List[Dict[str, Any]]) -> List[float]:
    """Runs preprocessing and predicts demand for a list of input records."""
    if model is None or scaler is None or encoder is None:
        raise ValueError("Model components are not loaded on server.")
        
    encoded_records = []
    
    for r in records:
        # Create a copy to edit
        item = r.copy()
        
        # Ensure store_id is integer
        item['store_id'] = parse_store_id(item.get('store_id', 1))
        
        # Parse date components if present, or use defaults
        date_str = item.get('date')
        if date_str:
            date_obj = pd.to_datetime(date_str)
            item['month'] = date_obj.month
            item['day_of_week'] = date_obj.dayofweek
            item['day'] = date_obj.day
            
        # Check if all required features are present
        missing_cols = []
        for col in FEATURE_COLS:
            if col in ['month', 'day_of_week', 'day']:
                # These can be derived from 'date' or provided directly
                if col not in item and 'date' not in item:
                    missing_cols.append(col)
            else:
                if col not in item or item[col] is None:
                    missing_cols.append(col)
                    
        if missing_cols:
            raise ValueError(f"Missing required features for prediction: {', '.join(missing_cols)}")

        # Categorical encoding
        categorical_cols = ['sku', 'size', 'color', 'category', 'sub_category', 'color_type', 'country', 'city']
        for col in categorical_cols:
            if col in encoder:
                item[col] = safe_label_transform(encoder[col], item[col])
                
        encoded_records.append(item)
        
    # Build dataframe with correct order
    df = pd.DataFrame(encoded_records)[FEATURE_COLS]
    
    # Scale features
    scaled_features = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled_features, columns=FEATURE_COLS)
    
    # Predict
    preds = model.predict(scaled_df)
    
    # Return rounded positive values
    return [round(max(0.0, float(p)), 2) for p in preds]

# ==========================================
# 3. FASTAPI ROUTERS (For EC2 / App Runner)
# ==========================================
class PredictionInput(BaseModel):
    records: List[Dict[str, Any]]

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Stateless Demand Forecasting ML API!",
        "health": "/health",
        "predict": "/predict"
    }

@app.get("/health")
def health():
    model_loaded = model is not None
    scaler_loaded = scaler is not None
    encoder_loaded = encoder is not None
    is_healthy = model_loaded and scaler_loaded and encoder_loaded
    return {
        "status": "healthy" if is_healthy else "degraded",
        "model_loaded": model_loaded,
        "scaler_loaded": scaler_loaded,
        "encoder_loaded": encoder_loaded
    }

@app.post("/predict")
def predict(payload: PredictionInput):
    """
    Accepts list of records:
    {
      "records": [
        { "store_id": "1", "sku": "SKU123", "date": "2026-06-24", ... }
      ]
    }
    Returns:
    {
      "status": "success",
      "predictions": [1.45, 2.30, ...]
    }
    """
    if not payload.records:
        raise HTTPException(status_code=400, detail="No records provided for prediction.")
    try:
        predictions = make_predictions(payload.records)
        return {
            "status": "success",
            "predictions": predictions
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

# ==========================================
# 4. AWS LAMBDA HANDLER ENTRY POINT
# ==========================================
def lambda_handler(event, context):
    """
    Standard AWS Lambda handler function.
    Handles inputs from API Gateway Proxy integration or direct invocation.
    """
    try:
        # Extract body
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {}) or event
            
        # Parse records
        records = body.get('records', [])
        
        # Fallback if request is a single record instead of a list of records
        if not records and ('sku' in body or 'date' in body or 'store_id' in body):
            records = [body]
            
        if not records:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'status': 'error',
                    'message': "Missing 'records' list in request body."
                })
            }
            
        predictions = make_predictions(records)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'status': 'success',
                'predictions': predictions
            })
        }
    except Exception as e:
        logger.error(f"Lambda execution error: {e}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'status': 'error',
                'message': f'System Error in Lambda: {str(e)}'
            })
        }
