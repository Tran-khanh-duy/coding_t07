import pyodbc

drivers = pyodbc.drivers()
print(f"Available Drivers: {drivers}")

server_names = [r"KHAIPOGBA", r"(local)", r".", r"localhost", r"127.0.0.1"]
database = "FaceAttendanceDB"
driver_to_use = "ODBC Driver 17 for SQL Server"

if driver_to_use not in drivers:
    print(f"WARNING: {driver_to_use} not found. Using the first available SQL Server driver.")
    sql_drivers = [d for d in drivers if "SQL Server" in d]
    if sql_drivers:
        driver_to_use = sql_drivers[0]
        print(f"Using driver: {driver_to_use}")
    else:
        print("ERROR: No SQL Server driver found.")
        exit(1)

for server in server_names:
    conn_str = f"DRIVER={{{driver_to_use}}};SERVER={server};DATABASE={database};Trusted_Connection=yes;TrustServerCertificate=yes;"
    print(f"Testing connection to {server}...")
    try:
        conn = pyodbc.connect(conn_str, timeout=5)
        print(f"SUCCESS: Connected to {server}!")
        conn.close()
        break
    except Exception as e:
        print(f"FAILED: Error connecting to {server}: {e}")
