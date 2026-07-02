import json
from lambda_function import lambda_handler

def run_test_case(name, event):
    print(f"\n==========================================")
    print(f" RUNNING TEST: {name}")
    print(f"==========================================")
    
    # Invoke lambda function
    response = lambda_handler(event, None)
    
    # Print status and headers
    print(f"HTTP Status Code: {response.get('statusCode')}")
    print(f"Headers: {response.get('headers')}")
    
    # Parse and format the body
    body_str = response.get('body', '{}')
    try:
        body_json = json.loads(body_str)
        print("Response Body:")
        print(json.dumps(body_json, indent=4))
    except Exception as e:
        print("Raw Response Body (Safe ASCII):")
        print(body_str.encode('ascii', 'backslashreplace').decode('ascii'))
        print(f"Error parsing body: {e}")
        
    return response

if __name__ == "__main__":
    # Test Case 1: Standard API Gateway Proxy Request (With real SKU)
    event_1 = {
        "body": json.dumps({
            "store_id": "STORE_001",
            "sku": "CHAC10010--",
            "date": "2026-06-24",
            "total_discount_avg": 0.15
        })
    }
    
    # Test Case 2: Missing fields (should fail under strict validation due to missing required fields)
    event_2 = {
        "body": json.dumps({
            "date": "2026-10-31"
        })
    }
    
    # Test Case 3: Direct Lambda Invocation Payload (not stringified body)
    event_3 = {
        "store_id": "2",
        "sku": "CHAC10010--",
        "date": "2026-07-04",
        "total_discount_avg": 0.25
    }
    
    # Test Case 4: Invalid date format (Triggering error handling)
    event_4 = {
        "body": json.dumps({
            "date": "not-a-valid-date"
        })
    }
    
    run_test_case("1. API Gateway Proxy Request (Real SKU)", event_1)
    run_test_case("2. Missing Fields (Expected Error)", event_2)
    run_test_case("3. Direct Invocation Format", event_3)
    run_test_case("4. Invalid Input (Error Handling)", event_4)
