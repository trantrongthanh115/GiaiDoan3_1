import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def run_api_test(name, endpoint, method="POST", payload=None):
    print(f"\n==========================================")
    print(f" RUNNING API TEST: {name}")
    print(f"==========================================")
    
    if method == "GET":
        response = client.get(endpoint)
    else:
        response = client.post(endpoint, json=payload)
        
    print(f"HTTP Status Code: {response.status_code}")
    print(f"Response JSON:")
    try:
        print(json.dumps(response.json(), indent=4))
    except Exception as e:
        print(f"Raw Response Content: {response.content}")
        print(f"Error parsing JSON: {e}")
        
    return response

if __name__ == "__main__":
    # Test Case 1: Health check
    run_api_test("1. GET /health Check", "/health", method="GET")
    
    # Test Case 2: Predict single (with standard details)
    single_payload = {
        "store_id": "STORE_001",
        "sku": "CHAC10010--",
        "date": "2026-06-24",
        "total_discount_avg": 0.15
    }
    run_api_test("2. POST /predict Single Item", "/predict", method="POST", payload=single_payload)
    
    # Test Case 3: Predict single with missing features (should fail with HTTP 400 under strict validation)
    missing_payload = {
        "store_id": "2",
        "sku": "MAT-716-L-WHITE",
        "date": "2026-10-31"
    }
    run_api_test("3. POST /predict Missing Features Error (Expected 400)", "/predict", method="POST", payload=missing_payload)
    
    # Test Case 4: Batch Prediction
    batch_payload = {
        "items": [
            {
                "store_id": "1",
                "sku": "CHAC10010--",
                "date": "2026-07-01",
                "total_discount_avg": 0.10
            },
            {
                "store_id": "STORE_002",
                "sku": "MASU485-M-",
                "date": "2026-07-02"
            }
        ]
    }
    run_api_test("4. POST /predict/batch Multi-items", "/predict/batch", method="POST", payload=batch_payload)
    
    # Test Case 5: Validation Error (Invalid Date Format)
    invalid_payload = {
        "store_id": "1",
        "sku": "CHAC10010--",
        "date": "invalid-date-format"
    }
    run_api_test("5. POST /predict Validation Error", "/predict", method="POST", payload=invalid_payload)

    # Test Case 6: Predict single passing all features directly
    all_features_payload = {
        "store_id": "1",
        "product_id": 8090,
        "sku": "CHAC10010--",
        "size": "40",
        "color": "BLACK",
        "date": "2026-06-24",
        "total_lifespan": 120.0,
        "s_total_lifespan": 180.0,
        "sku_store_coverage": 31.0,
        "product_store_coverage": 34.0,
        "s_days_active": 86.0,
        "s_selling_days_count": 42.0,
        "s_sales_velocity": 0.45,
        "s2_days_active": 67.0,
        "s2_selling_days_count": 4.0,
        "s2_sales_velocity": 0.06,
        "avg_usd_price": 51.3,
        "total_discount_avg": 0.10,
        "lag_1d": 1.0,
        "lag_7d": 0.0,
        "rolling_mean_7d": 0.8,
        "rolling_std_7d": 0.15,
        "category": "Feminine",
        "sub_category": "Dresses and Jumpsuits",
        "color_type": "Cor Unica",
        "country": "France",
        "city": "Paris",
        "num_distinct_products": 16500,
        "num_distinct_skus": 44000
    }
    run_api_test("6. POST /predict with All Feature Columns", "/predict", method="POST", payload=all_features_payload)

    # Test Case 7: Predict forecast with daily, weekly, monthly aggregations
    forecast_payload = {
        "items": [
            {
                "store_id": "1",
                "sku": "CHAC10010--",
                "date": "2026-06-01", # Week 23, Month 6
                "total_discount_avg": 0.10,
                "avg_usd_price": 50.0
            },
            {
                "store_id": "1",
                "sku": "CHAC10010--",
                "date": "2026-06-02", # Week 23, Month 6
                "total_discount_avg": 0.10,
                "avg_usd_price": 50.0
            },
            {
                "store_id": "1",
                "sku": "CHAC10010--",
                "date": "2026-06-15", # Week 25, Month 6
                "total_discount_avg": 0.15,
                "avg_usd_price": 50.0
            },
            {
                "store_id": "1",
                "sku": "CHAC10010--",
                "date": "2026-07-01", # Week 27, Month 7
                "total_discount_avg": 0.05,
                "avg_usd_price": 50.0
            }
        ]
    }
    run_api_test("7. POST /predict/forecast Aggregates (Day, Week, Month)", "/predict/forecast", method="POST", payload=forecast_payload)


