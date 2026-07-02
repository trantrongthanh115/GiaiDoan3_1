import os
import joblib
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from app.config import settings

logger = logging.getLogger("app.predictor")

# Base directory for absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class PredictorService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoder = None
        
        self.model_path = os.path.join(BASE_DIR, settings.MODEL_PATH)
        self.scaler_path = os.path.join(BASE_DIR, settings.SCALER_PATH)
        self.encoder_path = os.path.join(BASE_DIR, settings.ENCODER_PATH)
        
        self.load_models()
        
    def load_models(self):
        """Loads LightGBM model, standard scaler, and label encoders from pickle files."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded LightGBM model from {self.model_path}")
            else:
                logger.error(f"Model file not found at {self.model_path}")

            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Loaded Standard Scaler from {self.scaler_path}")
            else:
                logger.error(f"Scaler file not found at {self.scaler_path}")

            if os.path.exists(self.encoder_path):
                self.encoder = joblib.load(self.encoder_path)
                logger.info(f"Loaded Label Encoders from {self.encoder_path}")
            else:
                logger.error(f"Encoder file not found at {self.encoder_path}")
        except Exception as e:
            logger.error(f"Error loading models or preprocessors: {e}")
            raise e

    def safe_label_transform(self, label_encoder, value: Any) -> int:
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

    def parse_store_id(self, store_id_raw: str) -> int:
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

    def get_default_features(self) -> Dict[str, Any]:
        """Provides default fallback features for database misses."""
        return {
            'product_id': 8090,
            'size': '40',
            'color': 'BLACK',
            'total_lifespan': 120.0,
            's_total_lifespan': 180.0,
            'sku_store_coverage': 31.0,
            'product_store_coverage': 34.0,
            's_days_active': 86.0,
            's_selling_days_count': 42.0,
            's_sales_velocity': 0.45,
            's2_days_active': 67.0,
            's2_selling_days_count': 4.0,
            's2_sales_velocity': 0.06,
            'avg_usd_price': 51.3,
            'total_discount_avg': 0.10,
            'lag_1d': 1.0,
            'lag_7d': 0.0,
            'rolling_mean_7d': 0.8,
            'rolling_std_7d': 0.15,
            'category': 'Feminine',
            'sub_category': 'Dresses and Jumpsuits',
            'color_type': 'Cor Unica',
            'country': 'France',
            'city': 'Paris',
            'num_distinct_products': 16500,
            'num_distinct_skus': 44000
        }

    def predict(self, store_id_raw: str, sku: str, date_str: str, 
                db_features: Optional[Dict[str, Any]] = None,
                discount_override: Optional[float] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Runs the demand prediction.
        Returns:
            - predicted_quantity (float)
            - final features dictionary used in the model
        """
        if self.model is None or self.scaler is None or self.encoder is None:
            raise ValueError("PredictorService not fully initialized. Check file paths.")

        # 1. Parse store_id and time features
        store_id = self.parse_store_id(store_id_raw)
        
        date_obj = pd.to_datetime(date_str)
        month = date_obj.month
        day_of_week = date_obj.dayofweek
        day = date_obj.day

        # 2. Merge database features or fallback defaults
        features = self.get_default_features()
        if db_features:
            features.update(db_features)
            
        # Apply discount override if specified in request
        if discount_override is not None:
            features['total_discount_avg'] = discount_override

        # Assemble basic input keys
        input_data = {
            'store_id': store_id,
            'sku': sku,
            'month': month,
            'day_of_week': day_of_week,
            'day': day
        }
        input_data.update(features)

        # 3. Categorical encoding
        encoded_data = input_data.copy()
        categorical_cols = ['sku', 'size', 'color', 'category', 'sub_category', 'color_type', 'country', 'city']
        for col in categorical_cols:
            if col in self.encoder:
                encoded_data[col] = self.safe_label_transform(self.encoder[col], input_data[col])

        # 4. Construct feature matrix in correct column order
        feature_cols = [
            'store_id', 'product_id', 'sku', 'size', 'color', 
            'total_lifespan', 's_total_lifespan', 'sku_store_coverage', 'product_store_coverage', 
            's_days_active', 's_selling_days_count', 's_sales_velocity', 
            's2_days_active', 's2_selling_days_count', 's2_sales_velocity', 
            'avg_usd_price', 'total_discount_avg', 'lag_1d', 'lag_7d', 
            'rolling_mean_7d', 'rolling_std_7d', 'category', 'sub_category', 
            'color_type', 'country', 'city', 'num_distinct_products', 'num_distinct_skus', 
            'month', 'day_of_week', 'day'
        ]
        
        features_df = pd.DataFrame([encoded_data])[feature_cols]

        # 5. Apply scaling
        scaled_features = self.scaler.transform(features_df)
        scaled_features_df = pd.DataFrame(scaled_features, columns=feature_cols)

        # 6. Predict
        prediction = self.model.predict(scaled_features_df)
        predicted_quantity = float(prediction[0])
        predicted_quantity = max(0.0, predicted_quantity) # Demand cannot be negative

        # Return prediction and the unencoded details used (excluding index mapping)
        return predicted_quantity, input_data

# Singleton instance
predictor_service = PredictorService()
