import os

class Settings:
    # App Settings
    APP_NAME: str = "Fashion Retail Demand Forecasting API"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    
    # DB Settings (RDS PostgreSQL)
    DB_HOST: str = os.getenv("DB_HOST", "fashion-rds.c7846wiue0od.ap-southeast-1.rds.amazonaws.com")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "fashiondb")
    DB_USER: str = os.getenv("DB_USER", "dbadmin")
    DB_PASS: str = os.getenv("DB_PASS", "Tung2004")
    
    # Model Settings
    MODEL_PATH: str = os.getenv("MODEL_PATH", "lightgbm_demand_model.pkl")
    SCALER_PATH: str = os.getenv("SCALER_PATH", "standard_scaler.pkl")
    ENCODER_PATH: str = os.getenv("ENCODER_PATH", "label_encoders.pkl")
    
    # Layer 2 prediction API URL (AWS Lambda/App Runner or local)
    LAMBDA_PREDICT_URL: str = os.getenv("LAMBDA_PREDICT_URL", "http://localhost:8080/predict")

settings = Settings()
