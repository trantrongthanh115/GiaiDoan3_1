import logging
import json
import urllib.request
import urllib.error
from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse, HealthResponse,
    ForecastResponse, DailyForecastResult, WeeklyForecastResult, MonthlyForecastResult
)
from app.database import fetch_features_from_db, fetch_static_features, save_weekly_forecast
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("app.main")

# Initialize FastAPI application
app = FastAPI(
    title="Global Fashion Retail Demand Forecasting API",
    description="Orchestrator API (Layer 1) built with FastAPI to coordinate RDS features and call the Stateless ML API.",
    version="1.0.0"
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# HELPER FUNCTIONS FOR INTER-MODULE AND LAYER CALLS
# ==========================================
def parse_store_id(store_id_raw: str) -> int:
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

def check_layer2_health() -> bool:
    """Checks the health of the Layer 2 Stateless ML API."""
    url = settings.LAMBDA_PREDICT_URL.replace("/predict", "/health")
    try:
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=3) as response:
            res_body = response.read().decode('utf-8')
            res_json = json.loads(res_body)
            return res_json.get("status") == "healthy"
    except Exception as e:
        logger.warning(f"Layer 2 health check failed: {e}")
        return False

def call_layer2_predict(records: List[dict]) -> List[float]:
    """Calls Layer 2 Stateless ML API to predict demand for a list of records."""
    url = settings.LAMBDA_PREDICT_URL
    payload = {"records": records}
    data = json.dumps(payload).encode('utf-8')
    
    req = urllib.request.Request(
        url,
        data=data,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            res_body = response.read().decode('utf-8')
            res_json = json.loads(res_body)
            if res_json.get("status") == "success":
                return res_json.get("predictions", [])
            else:
                raise ValueError(f"Layer 2 error: {res_json.get('message')}")
    except Exception as e:
        logger.error(f"Error calling Layer 2 ML API: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to connect or retrieve predictions from Layer 2 ML Service. Details: {str(e)}"
        )

REQUIRED_FEATURES = [
    'product_id', 'size', 'color', 'total_lifespan', 's_total_lifespan', 
    'sku_store_coverage', 'product_store_coverage', 's_days_active', 
    's_selling_days_count', 's_sales_velocity', 's2_days_active', 
    's2_selling_days_count', 's2_sales_velocity', 'avg_usd_price', 
    'total_discount_avg', 'lag_1d', 'lag_7d', 'rolling_mean_7d', 'rolling_std_7d', 
    'category', 'sub_category', 'color_type', 'country', 'city', 
    'num_distinct_products', 'num_distinct_skus'
]

def get_features_payload(store_id_raw: str, sku: str, date_str: str, 
                         discount_override: Optional[float] = None,
                         all_request_features: Optional[dict] = None) -> dict:
    """Fetches and aggregates all feature variables needed by the ML model. Raises error if features are missing."""
    store_id = parse_store_id(store_id_raw)
    
    # 1. Fetch features from DB
    db_features = fetch_features_from_db(
        store_id=store_id,
        sku=sku,
        date_str=date_str
    )
    
    # Fallback to static table lookup if daily feature store has no record
    if not db_features:
        logger.info(f"Precalculated features for store {store_id} and SKU {sku} not found. Attempting static tables fallback.")
        db_features = fetch_static_features(
            store_id=store_id,
            sku=sku
        )
        
    if not db_features:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Features not found in database for store_id={store_id_raw} and sku={sku}."
        )

    # Construct final payload format
    payload = {
        'store_id': store_id_raw,
        'sku': sku,
        'date': date_str
    }
    
    # Merge DB features
    payload.update(db_features)
    
    # Override with request features if provided
    if all_request_features:
        for k, v in all_request_features.items():
            if k not in ['store_id', 'sku', 'date'] and v is not None:
                payload[k] = v
                
    # Apply discount override
    if discount_override is not None:
        payload['total_discount_avg'] = discount_override

    # Validate that all required model features are present
    missing_features = [f for f in REQUIRED_FEATURES if f not in payload or payload[f] is None]
    if missing_features:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required model features for store_id={store_id_raw}, sku={sku}: {', '.join(missing_features)}"
        )

    return payload


# ==========================================
# FASTAPI ENDPOINTS
# ==========================================
@app.get("/")
def read_root():
    """Welcome message and documentation link."""
    return {
        "message": "Welcome to the Layer 1 Global Fashion Retail Demand Forecasting Orchestrator API!",
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def health_check():
    """Verify system health by checking connection to Layer 2."""
    l2_healthy = check_layer2_health()
    return HealthResponse(
        status="healthy" if l2_healthy else "degraded",
        model_loaded=l2_healthy,
        scaler_loaded=l2_healthy,
        encoder_loaded=l2_healthy
    )

@app.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
def predict_single(request: PredictRequest):
    """
    Predict daily quantity demand for a single store, SKU, and date combination.
    Queries the database first to find features, then forwards them to Layer 2 Stateless API.
    """
    try:
        # 1. Fetch and aggregate all features in Layer 1
        request_features = request.dict(exclude_unset=True)
        features_payload = get_features_payload(
            store_id_raw=request.store_id,
            sku=request.sku,
            date_str=request.date,
            discount_override=request.total_discount_avg,
            all_request_features=request_features
        )

        # 2. Call Layer 2 (Stateless ML API) to run prediction
        predictions = call_layer2_predict([features_payload])
        predicted_qty = predictions[0]

        return PredictResponse(
            status="success",
            predicted_quantity=predicted_qty,
            input_received=request_features,
            features_used=features_payload
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in predict_single: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction orchestrator failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictResponse, status_code=status.HTTP_200_OK)
def predict_batch(request: BatchPredictRequest):
    """
    Predict demand for multiple items concurrently.
    Gathers features in Layer 1, sends them as a batch to Layer 2, and returns predictions.
    """
    if not request.items:
        raise HTTPException(status_code=400, detail="Empty items list in request.")
        
    records_to_predict = []
    items_meta = []
    
    # 1. Gather features for each item
    for item in request.items:
        try:
            item_features = item.dict(exclude_unset=True)
            payload = get_features_payload(
                store_id_raw=item.store_id,
                sku=item.sku,
                date_str=item.date,
                discount_override=item.total_discount_avg,
                all_request_features=item_features
            )
            records_to_predict.append(payload)
            items_meta.append((True, item_features))
        except Exception as e:
            logger.error(f"Error preparing features for batch item {item.sku}: {e}")
            items_meta.append((False, item.dict(exclude_unset=True), str(e)))

    # 2. Send valid records to Layer 2 in a single batch
    valid_records = [r for success, r in zip(items_meta, records_to_predict) if success[0]]
    predictions = []
    if valid_records:
        try:
            predictions = call_layer2_predict(valid_records)
        except Exception as e:
            logger.error(f"Batch prediction call failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed calling ML service: {str(e)}"
            )

    # 3. Assemble response list
    results = []
    pred_idx = 0
    
    for meta in items_meta:
        if meta[0]:
            results.append(
                PredictResponse(
                    status="success",
                    predicted_quantity=predictions[pred_idx],
                    input_received=meta[1],
                    features_used=records_to_predict[pred_idx]
                )
            )
            pred_idx += 1
        else:
            results.append(
                PredictResponse(
                    status="error",
                    predicted_quantity=0.0,
                    input_received=meta[1],
                    features_used={"error": meta[2]}
                )
            )
            
    return BatchPredictResponse(
        status="success",
        predictions=results
    )

@app.post("/predict/forecast", response_model=ForecastResponse, status_code=status.HTTP_200_OK)
def predict_forecast(request: BatchPredictRequest):
    """
    Predict demand by day, and aggregate predictions by week and month.
    Gathers features, calls Layer 2, aggregates weekly/monthly predictions, 
    and saves weekly aggregated predictions to RDS PostgreSQL database.
    """
    if not request.items:
        raise HTTPException(status_code=400, detail="Empty items list in request.")

    daily_results = []
    weekly_groups = {}
    monthly_groups = {}
    
    records_to_predict = []
    valid_items = []
    
    # 1. Gather features for all items
    for item in request.items:
        item_features = item.dict(exclude_unset=True)
        payload = get_features_payload(
            store_id_raw=item.store_id,
            sku=item.sku,
            date_str=item.date,
            discount_override=item.total_discount_avg,
            all_request_features=item_features
        )
        records_to_predict.append(payload)
        valid_items.append(item)

    # 2. Call Layer 2 for all valid features in batch
    predictions = []
    if records_to_predict:
        predictions = call_layer2_predict(records_to_predict)

    # 3. Aggregate daily predictions by week and month
    for item, payload, pred_qty in zip(valid_items, records_to_predict, predictions):
        # Daily result
        daily_results.append(
            DailyForecastResult(
                store_id=item.store_id,
                sku=item.sku,
                date=item.date,
                predicted_quantity=pred_qty
            )
        )
        
        # Parse date
        date_obj = pd.to_datetime(item.date)
        year = int(date_obj.year)
        month = int(date_obj.month)
        week = int(date_obj.isocalendar()[1])
        
        product_id = payload.get('product_id', 8090)
        
        # Weekly aggregation key: (store_id, sku, product_id, year, week)
        w_key = (item.store_id, item.sku, product_id, year, week)
        weekly_groups[w_key] = weekly_groups.get(w_key, 0.0) + pred_qty
        
        # Monthly aggregation key: (store_id, sku, year, month)
        m_key = (item.store_id, item.sku, year, month)
        monthly_groups[m_key] = monthly_groups.get(m_key, 0.0) + pred_qty

    # 4. Save weekly predictions back to RDS database
    for w_key, total_qty in weekly_groups.items():
        store_id, sku, product_id, year, week = w_key
        # Save to DB
        db_success = save_weekly_forecast(
            store_id=store_id,
            sku=sku,
            product_id=str(product_id),
            year=year,
            week=week,
            predicted_qty=round(total_qty, 2),
            model_version="lightgbm_v1"
        )
        if not db_success:
            logger.warning(f"Could not save weekly forecast to DB for key: {w_key}")

    # 5. Format response lists
    weekly_results = [
        WeeklyForecastResult(
            store_id=k[0],
            sku=k[1],
            year=k[3],
            week=k[4],
            predicted_quantity=round(val, 2)
        )
        for k, val in weekly_groups.items()
    ]
    
    monthly_results = [
        MonthlyForecastResult(
            store_id=k[0],
            sku=k[1],
            year=k[2],
            month=k[3],
            predicted_quantity=round(val, 2)
        )
        for k, val in monthly_groups.items()
    ]
    
    return ForecastResponse(
        status="success",
        daily=daily_results,
        weekly=weekly_results,
        monthly=monthly_results
    )
