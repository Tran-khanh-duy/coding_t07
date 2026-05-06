
from database.connection import get_db
try:
    rows = get_db().execute("SELECT MaToa, TenToa FROM ToaNha")
    print("Buildings in DB:", rows)
    rows_lop = get_db().execute("SELECT IDLop, TenLop FROM lop")
    print("Classes in DB:", rows_lop)
except Exception as e:
    print("Error:", e)
