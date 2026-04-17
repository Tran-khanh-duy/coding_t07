import pyodbc
import sys

def test_connection(server, database, driver):
    conn_str = (
        f"DRIVER={{{driver}}};SERVER={server};"
        f"DATABASE={database};Trusted_Connection=yes;"
        f"TrustServerCertificate=yes;"
    )
    print(f"Testing conection with: {conn_str}")
    try:
        conn = pyodbc.connect(conn_str, timeout=5)
        print("Success!")
        conn.close()
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False

if __name__ == "__main__":
    server = "."
    database = "FaceAttendanceDB"
    driver = "ODBC Driver 17 for SQL Server"
    
    print("--- Test 1: Default ('.') ---")
    test_connection(".", "master", driver)
    
    print("\n--- Test 2: Localhost ---")
    test_connection("localhost", "master", driver)
    
    print("\n--- Test 3: (local) ---")
    test_connection("(local)", "master", driver)

    print("\n--- Test 4: Full DB Check ---")
    test_connection(".", "FaceAttendanceDB", driver)
