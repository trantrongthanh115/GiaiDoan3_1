from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime

class PredictRequest(BaseModel):
    store_id: str = Field(..., description="ID of the store (e.g., '1', 'STORE_001')", example="1")
    product_id: Optional[int] = Field(None, description="Product ID", example=10010)
    sku: str = Field(..., description="SKU code of the product (e.g., 'CHAC10010--')", example="CHAC10010--")
    size: Optional[str] = Field(None, description="Size of the product (e.g., 'M', 'L')", example="M")
    color: Optional[str] = Field(None, description="Color of the product (e.g., 'BLACK')", example="BLACK")
    date: str = Field(..., description="Target prediction date in YYYY-MM-DD format", example="2026-06-24")
    
    # Model features
    total_lifespan: Optional[float] = Field(None, example=120.0)
    s_total_lifespan: Optional[float] = Field(None, example=180.0)
    sku_store_coverage: Optional[float] = Field(None, example=31.0)
    product_store_coverage: Optional[float] = Field(None, example=34.0)
    s_days_active: Optional[float] = Field(None, example=86.0)
    s_selling_days_count: Optional[float] = Field(None, example=42.0)
    s_sales_velocity: Optional[float] = Field(None, example=0.45)
    s2_days_active: Optional[float] = Field(None, example=67.0)
    s2_selling_days_count: Optional[float] = Field(None, example=4.0)
    s2_sales_velocity: Optional[float] = Field(None, example=0.06)
    avg_usd_price: Optional[float] = Field(None, example=50.0)
    total_discount_avg: Optional[float] = Field(None, description="Optional override for discount average (0.0 to 1.0)", example=0.15)
    lag_1d: Optional[float] = Field(None, example=1.0)
    lag_7d: Optional[float] = Field(None, example=0.0)
    rolling_mean_7d: Optional[float] = Field(None, example=0.8)
    rolling_std_7d: Optional[float] = Field(None, example=0.15)
    category: Optional[str] = Field(None, example="Children")
    sub_category: Optional[str] = Field(None, example="Accessories")
    color_type: Optional[str] = Field(None, example="Cor Unica")
    country: Optional[str] = Field(None, example="France")
    city: Optional[str] = Field(None, example="Paris")
    num_distinct_products: Optional[int] = Field(None, example=16500)
    num_distinct_skus: Optional[int] = Field(None, example=44000)

    @validator('date')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v

    @validator('total_discount_avg')
    def validate_discount(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Discount average must be between 0.0 and 1.0")
        return v

class BatchPredictRequest(BaseModel):
    items: List[PredictRequest]

class PredictResponse(BaseModel):
    status: str = Field(..., example="success")
    predicted_quantity: float = Field(..., description="Forecasted daily quantity demand", example=1.19)
    input_received: dict = Field(..., description="The original request input features echoed back")
    features_used: Optional[dict] = Field(None, description="Features used by the model for the prediction")

class BatchPredictResponse(BaseModel):
    status: str = Field(..., example="success")
    predictions: List[PredictResponse]

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    scaler_loaded: bool = Field(..., example=True)
    encoder_loaded: bool = Field(..., example=True)

class DailyForecastResult(BaseModel):
    store_id: str = Field(..., example="1")
    sku: str = Field(..., example="CHAC10010--")
    date: str = Field(..., example="2026-06-24")
    predicted_quantity: float = Field(..., example=1.19)

class WeeklyForecastResult(BaseModel):
    store_id: str = Field(..., example="1")
    sku: str = Field(..., example="CHAC10010--")
    year: int = Field(..., example=2026)
    week: int = Field(..., example=26)
    predicted_quantity: float = Field(..., example=8.33)

class MonthlyForecastResult(BaseModel):
    store_id: str = Field(..., example="1")
    sku: str = Field(..., example="CHAC10010--")
    year: int = Field(..., example=2026)
    month: int = Field(..., example=6)
    predicted_quantity: float = Field(..., example=35.70)

class ForecastResponse(BaseModel):
    status: str = Field(..., example="success")
    daily: List[DailyForecastResult]
    weekly: List[WeeklyForecastResult]
    monthly: List[MonthlyForecastResult]

