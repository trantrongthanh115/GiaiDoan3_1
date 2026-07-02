import json
import os
import joblib
import pandas as pd
import numpy as np

# Connection details for AWS RDS PostgreSQL
DB_HOST = "fashion-rds.c7846wiue0od.ap-southeast-1.rds.amazonaws.com"
DB_PORT = "5432"
DB_NAME = "fashiondb"
DB_USER = "dbadmin"
DB_PASS = "Tung2004"

# ==========================================
# 1. LOAD MÔ HÌNH THẬT KHI LAMBDA KHỞI ĐỘNG
# ==========================================
model = None
scaler = None
encoder = None

try:
    # Load các tệp mô hình và tiền xử lý thật
    model = joblib.load('lightgbm_demand_model.pkl')
    scaler = joblib.load('standard_scaler.pkl')
    encoder = joblib.load('label_encoders.pkl')
    print("Successfully loaded model and preprocessors!")
except Exception as e:
    print(f"Model load error: {str(e)}")

# Safe label transformation helper to handle unseen labels and mismatches
def safe_label_transform(label_encoder, value):
    classes_list = label_encoder.classes_
    
    # Normalize value to string if encoder classes are strings
    if len(classes_list) > 0 and isinstance(classes_list[0], str):
        value_str = str(value)
        if value_str in classes_list:
            return int(label_encoder.transform([value_str])[0])
        # Case fallback check
        if value in classes_list:
            return int(label_encoder.transform([value])[0])
    else:
        if value in classes_list:
            return int(label_encoder.transform([value])[0])
            
    # Fallback to the first class if label is unseen
    fallback_val = classes_list[0]
    return int(label_encoder.transform([fallback_val])[0])

# Default mock features for local sandbox fallback when DB is unreachable
DEFAULT_FEATURES = {
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

# ==========================================
# 2. HÀM XỬ LÝ CHÍNH KHI CÓ REQUEST GỬI ĐẾN
# ==========================================
def lambda_handler(event, context):
    try:
        # a. Nhận dữ liệu đầu vào từ API Gateway
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {}) or event
            
        store_id_raw = body.get('store_id', '1') # Default to store 1
        sku = body.get('sku', 'SKU_999')
        date_str = body.get('date', '2026-06-24')
        total_discount_avg_req = body.get('total_discount_avg') # Optional override
        
        # Parse store_id to integer (since it is numerical in the model)
        # Handle formats like 'STORE_001' or '1'
        if isinstance(store_id_raw, str):
            if 'STORE_' in store_id_raw.upper():
                store_id = int(store_id_raw.upper().replace('STORE_', ''))
            else:
                try:
                    store_id = int(store_id_raw)
                except ValueError:
                    store_id = 1
        else:
            store_id = int(store_id_raw)
            
        # Parse date
        date_obj = pd.to_datetime(date_str)
        month = date_obj.month
        day_of_week = date_obj.dayofweek
        day = date_obj.day
        
        db_features = None
        
        # Check if we are running in AWS Lambda
        is_local = 'AWS_LAMBDA_FUNCTION_NAME' not in os.environ
        
        # Connect to DB and fetch precalculated features
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASS,
                connect_timeout=3
            )
            cur = conn.cursor()
            
            # Query features for the given store_id and sku
            query = """
                SELECT 
                    product_id, size, color, total_lifespan, s_total_lifespan, 
                    sku_store_coverage, product_store_coverage, s_days_active, 
                    s_selling_days_count, s_sales_velocity, s2_days_active, 
                    s2_selling_days_count, s2_sales_velocity, avg_usd_price, 
                    total_discount_avg, lag_1d, lag_7d, rolling_mean_7d, rolling_std_7d, 
                    category, sub_category, color_type, country, city, 
                    num_distinct_products, num_distinct_skus
                FROM final_daily
                WHERE store_id = %s AND sku = %s
                ORDER BY ABS(date::date - %s::date)
                LIMIT 1
            """
            cur.execute(query, (store_id, sku, date_str))
            row = cur.fetchone()
            cur.close()
            conn.close()
            
            if row:
                db_features = {
                    'product_id': int(row[0]),
                    'size': str(row[1]),
                    'color': str(row[2]),
                    'total_lifespan': float(row[3]),
                    's_total_lifespan': float(row[4]),
                    'sku_store_coverage': float(row[5]),
                    'product_store_coverage': float(row[6]),
                    's_days_active': float(row[7]),
                    's_selling_days_count': float(row[8]),
                    's_sales_velocity': float(row[9]),
                    's2_days_active': float(row[10]),
                    's2_selling_days_count': float(row[11]),
                    's2_sales_velocity': float(row[12]),
                    'avg_usd_price': float(row[13]),
                    'total_discount_avg': float(row[14]),
                    'lag_1d': float(row[15]),
                    'lag_7d': float(row[16]),
                    'rolling_mean_7d': float(row[17]),
                    'rolling_std_7d': float(row[18]),
                    'category': str(row[19]),
                    'sub_category': str(row[20]),
                    'color_type': str(row[21]),
                    'country': str(row[22]),
                    'city': str(row[23]),
                    'num_distinct_products': int(row[24]),
                    'num_distinct_skus': int(row[25])
                }
            else:
                raise ValueError(f"No features found in DB for store_id={store_id} and sku={sku}")
                
        except Exception as db_err:
            print(f"Database/Feature error: {str(db_err)}")
            raise db_err
                
        # Override discount if passed in request
        if total_discount_avg_req is not None:
            db_features['total_discount_avg'] = float(total_discount_avg_req)
            
        # Assemble all inputs
        inputs = {
            'store_id': store_id,
            'sku': sku,
            'month': month,
            'day_of_week': day_of_week,
            'day': day
        }
        inputs.update(db_features)
        
        # Check if all required features are present
        required_cols = [
            'store_id', 'product_id', 'sku', 'size', 'color', 
            'total_lifespan', 's_total_lifespan', 'sku_store_coverage', 'product_store_coverage', 
            's_days_active', 's_selling_days_count', 's_sales_velocity', 
            's2_days_active', 's2_selling_days_count', 's2_sales_velocity', 
            'avg_usd_price', 'total_discount_avg', 'lag_1d', 'lag_7d', 
            'rolling_mean_7d', 'rolling_std_7d', 'category', 'sub_category', 
            'color_type', 'country', 'city', 'num_distinct_products', 'num_distinct_skus', 
            'month', 'day_of_week', 'day'
        ]
        missing_cols = [col for col in required_cols if col not in inputs or inputs[col] is None]
        if missing_cols:
            raise ValueError(f"Missing required features for prediction: {', '.join(missing_cols)}")

        # Categorical Encoding
        encoded_inputs = inputs.copy()
        categorical_cols = ['sku', 'size', 'color', 'category', 'sub_category', 'color_type', 'country', 'city']
        for col in categorical_cols:
            if encoder and col in encoder:
                encoded_inputs[col] = safe_label_transform(encoder[col], inputs[col])
                
        # Ensure correct column order for model/scaler
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
        
        features_df = pd.DataFrame([encoded_inputs])[feature_cols]
        
        # Scaling
        if scaler:
            scaled_features = scaler.transform(features_df)
        else:
            scaled_features = features_df.to_numpy()
            
        # Re-create DataFrame with column names to suppress LightGBM UserWarning
        scaled_features_df = pd.DataFrame(scaled_features, columns=feature_cols)
            
        # Prediction
        if model:
            prediction = model.predict(scaled_features_df)
            predicted_quantity = float(prediction[0])
            predicted_quantity = max(0.0, predicted_quantity) # Demand cannot be negative
        else:
            raise ValueError("Model not loaded.")
            
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'status': 'success',
                'message': 'Dự báo thành công (Mô hình LightGBM thật)',
                'input_received': {
                    'store_id': store_id_raw,
                    'sku': sku,
                    'date': date_str,
                    'total_discount_avg': total_discount_avg_req
                },
                'predicted_quantity': round(predicted_quantity, 2)
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'status': 'error',
                'message': f'Lỗi hệ thống Lambda: {str(e)}'
            })
        }
