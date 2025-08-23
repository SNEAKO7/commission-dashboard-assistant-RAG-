import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

print("Testing database connection...")
print(f"Host: {os.getenv('DB_HOST')}")
print(f"Database: {os.getenv('DB_NAME')}")
print(f"User: {os.getenv('DB_USER')}")
print(f"Password: {'*' * len(os.getenv('DB_PASSWORD', ''))}")

try:
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=int(os.getenv('DB_PORT', ' ')),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    print("✅ Database connection successful!")
    
    # Test query
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM plan_master")
        count = cur.fetchone()[0]
        print(f"Found {count} records in plan_master table")
    
    conn.close()
    
except Exception as e:
    print(f"❌ Database connection failed: {e}")
