import pyodbc

def test_conn(server, database="master"):
    driver = "ODBC Driver 17 for SQL Server"
    conn_str = f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};Trusted_Connection=yes;TrustServerCertificate=yes;"
    try:
        conn = pyodbc.connect(conn_str, timeout=3)
        print(f"SUCCESS: {server} -> {database}")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sys.databases")
        dbs = [row[0] for row in cursor.fetchall()]
        print(f"Databases on {server}: {dbs}")
        conn.close()
        return True
    except Exception as e:
        print(f"FAILED: {server} -> {database}: {e}")
        return False

servers = [r"KHAIPOGBA", r".", r"(local)", r"localhost", r"127.0.0.1"]
print("Testing various server names...")
for s in servers:
    test_conn(s)

print("\nTesting specific database...")
test_conn(r"KHAIPOGBA", "FaceAttendanceDB")
test_conn(r".", "FaceAttendanceDB")
