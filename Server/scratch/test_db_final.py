import pyodbc
import sys
import os
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from config import db_config
    print(f"Loaded config: Server={db_config.server}, DB={db_config.database}")
except Exception as e:
    print(f"Failed to load config: {e}")
    sys.exit(1)

def test_connection():
    conn_str = db_config.connection_string
    print(f"Testing connection with config string: {conn_str}")
    try:
        conn = pyodbc.connect(conn_str, timeout=5)
        print("RESULT: Connection SUCCESS!")
        
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 1 camera_name FROM Cameras")
        row = cursor.fetchone()
        if row:
            print(f"RESULT: Query SUCCESS! (Found camera: {row[0]})")
        else:
            print("RESULT: Query SUCCESS, but no cameras found in DB.")
            
        conn.close()
        return True
    except Exception as e:
        print(f"RESULT: Connection FAILED: {e}")
        return False

if __name__ == "__main__":
    test_connection()
