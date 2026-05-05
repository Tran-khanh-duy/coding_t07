import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import get_db
from config import db_config

print(f"Connecting to: {db_config.host}:{db_config.port} / {db_config.database}")

try:
    db = get_db()
    rows = db.execute("SELECT session_id, status, subject_name FROM AttendanceSessions ORDER BY session_id DESC LIMIT 5")
    print("\nLATEST SESSIONS:")
    for r in rows:
        print(f"ID: {r[0]} | Status: {r[1]} | Subject: {r[2]}")
except Exception as e:
    print(f"Error: {e}")
