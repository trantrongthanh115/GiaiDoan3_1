import psycopg2
import logging
from typing import Dict, Any, Optional
from app.config import settings

logger = logging.getLogger("app.database")

def get_db_connection():
    """Establishes connection to the AWS RDS PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            database=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASS,
            connect_timeout=3
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise e

def check_table_exists(table_name: str) -> bool:
    """Checks if a table exists in the database schema."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            );
        """, (table_name,))
        exists = cur.fetchone()[0]
        cur.close()
        return exists
    except Exception:
        return False
    finally:
        if conn:
            conn.close()

def fetch_features_from_db(store_id: int, sku: str, date_str: str) -> Optional[Dict[str, Any]]:
    """
    Attempts to fetch precalculated features from the final_daily table.
    Returns None if the table doesn't exist or no record is found.
    """
    if not check_table_exists("final_daily"):
        logger.warning("Table 'final_daily' does not exist in the database.")
        return None
        
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Query features for the given store_id and sku nearest to the date
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
        cur.execute(query, (str(store_id), sku, date_str))
        row = cur.fetchone()
        cur.close()
        
        if row:
            return {
                'product_id': int(row[0]) if row[0] is not None else 0,
                'size': str(row[1]) if row[1] is not None else 'M',
                'color': str(row[2]) if row[2] is not None else 'BLACK',
                'total_lifespan': float(row[3]) if row[3] is not None else 120.0,
                's_total_lifespan': float(row[4]) if row[4] is not None else 180.0,
                'sku_store_coverage': float(row[5]) if row[5] is not None else 30.0,
                'product_store_coverage': float(row[6]) if row[6] is not None else 30.0,
                's_days_active': float(row[7]) if row[7] is not None else 90.0,
                's_selling_days_count': float(row[8]) if row[8] is not None else 40.0,
                's_sales_velocity': float(row[9]) if row[9] is not None else 0.4,
                's2_days_active': float(row[10]) if row[10] is not None else 60.0,
                's2_selling_days_count': float(row[11]) if row[11] is not None else 5.0,
                's2_sales_velocity': float(row[12]) if row[12] is not None else 0.05,
                'avg_usd_price': float(row[13]) if row[13] is not None else 50.0,
                'total_discount_avg': float(row[14]) if row[14] is not None else 0.0,
                'lag_1d': float(row[15]) if row[15] is not None else 0.0,
                'lag_7d': float(row[16]) if row[16] is not None else 0.0,
                'rolling_mean_7d': float(row[17]) if row[17] is not None else 0.0,
                'rolling_std_7d': float(row[18]) if row[18] is not None else 0.0,
                'category': str(row[19]) if row[19] is not None else 'Feminine',
                'sub_category': str(row[20]) if row[20] is not None else 'Dresses and Jumpsuits',
                'color_type': str(row[21]) if row[21] is not None else 'Cor Unica',
                'country': str(row[22]) if row[22] is not None else 'United States',
                'city': str(row[23]) if row[23] is not None else 'New York',
                'num_distinct_products': int(row[24]) if row[24] is not None else 17000,
                'num_distinct_skus': int(row[25]) if row[25] is not None else 50000
            }
        return None
    except Exception as e:
        logger.error(f"Error fetching features from DB: {e}")
        return None
    finally:
        if conn:
            conn.close()

def fetch_static_features(store_id: int, sku: str) -> Optional[Dict[str, Any]]:
    """
    Fetches static store and product attributes from raw tables as a fallback.
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 1. Fetch product info
        # Extract product_id from sku if possible (e.g. FECO131-M-BLUE -> 131)
        # Sku formats vary, we will try to look up by sku first, then fallback to parsing product_id
        product_info = None
        cur.execute("""
            SELECT product_id, size, color, category, sub_category, cost
            FROM products
            WHERE sku = %s
            LIMIT 1
        """, (sku,))
        row = cur.fetchone()
        
        if not row:
            # Try to parse product_id from SKU string (usually letters followed by numbers or inside the sku name)
            # E.g. "FECO131-M-BLUE" contains "131"
            import re
            match = re.search(r'\d+', sku)
            if match:
                parsed_id = match.group(0)
                cur.execute("""
                    SELECT product_id, size, color, category, sub_category, cost
                    FROM products
                    WHERE product_id = %s
                    LIMIT 1
                """, (parsed_id,))
                row = cur.fetchone()
                
        if row:
            color_val = row[2] if row[2] is not None else 'BLACK'
            product_info = {
                'product_id': int(row[0]) if row[0] is not None else 0,
                'size': str(row[1]) if row[1] is not None else 'M',
                'color': str(color_val),
                'category': str(row[3]) if row[3] is not None else 'Feminine',
                'sub_category': str(row[4]) if row[4] is not None else 'Coats and Blazers',
                'color_type': 'Multi Color' if color_val == 'None' or color_val is None else 'Cor Unica',
                'avg_usd_price': float(row[5]) if row[5] is not None else 50.0 # using cost/price as proxy if needed
            }
            
        # 2. Fetch store info
        store_info = None
        cur.execute("""
            SELECT country, city, num_employees
            FROM stores
            WHERE store_id = %s
            LIMIT 1
        """, (str(store_id),))
        row_store = cur.fetchone()
        
        if row_store:
            store_info = {
                'country': str(row_store[0]) if row_store[0] is not None else 'United States',
                'city': str(row_store[1]) if row_store[1] is not None else 'New York',
                'num_distinct_products': int(row_store[2]) * 1700 if row_store[2] is not None else 17000, # proxy size
                'num_distinct_skus': int(row_store[2]) * 5000 if row_store[2] is not None else 50000
            }
            
        cur.close()
        
        # Combine if we have information
        if product_info or store_info:
            res = {}
            if product_info:
                res.update(product_info)
            if store_info:
                res.update(store_info)
            return res
            
        return None
    except Exception as e:
        logger.error(f"Error fetching static features from DB: {e}")
        return None
    finally:
        if conn:
            conn.close()

def save_weekly_forecast(store_id: str, sku: str, product_id: str, year: int, week: int, predicted_qty: float, model_version: str = "lightgbm_v1") -> bool:
    """
    Saves or updates a weekly demand forecast in the demand_forecasts table.
    """
    if not check_table_exists("demand_forecasts"):
        logger.warning("Table 'demand_forecasts' does not exist in the database. Cannot save forecast.")
        return False
        
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if record already exists for this store, sku, year, week
        check_query = """
            SELECT forecast_id FROM demand_forecasts 
            WHERE store_id = %s AND sku = %s AND forecast_year = %s AND forecast_week = %s
        """
        cur.execute(check_query, (str(store_id), sku, year, week))
        row = cur.fetchone()
        
        if row:
            # Update existing prediction
            update_query = """
                UPDATE demand_forecasts 
                SET predicted_quantity = %s, model_version = %s, created_at = NOW()
                WHERE forecast_id = %s
            """
            cur.execute(update_query, (predicted_qty, model_version, row[0]))
            logger.info(f"Updated forecast in DB for store {store_id}, SKU {sku}, Year {year}, Week {week} to {predicted_qty}")
        else:
            # Insert new prediction
            insert_query = """
                INSERT INTO demand_forecasts (store_id, sku, product_id, forecast_year, forecast_week, predicted_quantity, model_version, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            """
            cur.execute(insert_query, (str(store_id), sku, str(product_id), year, week, predicted_qty, model_version))
            logger.info(f"Inserted forecast in DB for store {store_id}, SKU {sku}, Year {year}, Week {week} with {predicted_qty}")
            
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        logger.error(f"Error saving forecast to DB: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

