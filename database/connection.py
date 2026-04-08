#database/connection.py
import pyodbc
import threading
from contextlib import contextmanager
from typing import Optional
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import db_config


class DatabaseConnection:
    """
    Singleton connection manager cho SQL Server.
    Thread-safe: mỗi thread có 1 connection riêng (thread-local).
    """
    _instance: Optional["DatabaseConnection"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._conn_str = db_config.connection_string
        self._local    = threading.local()
        self._initialized = True
        logger.info("DatabaseConnection initialized")

    # ─── Internal ──────────────────────────────

    def _get_connection(self) -> pyodbc.Connection:
        """Lấy connection của thread hiện tại, tạo mới nếu chưa có."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            try:
                conn = pyodbc.connect(
                    self._conn_str,
                    autocommit=False,
                    timeout=30,
                )
                # FIX v3: Không override decoding — để pyodbc tự dùng UTF-16LE
                # mặc định của Windows cho NVARCHAR. Gọi setdecoding(SQL_WCHAR,'utf-8')
                # sẽ gây UnicodeDecodeError với tiếng Việt vì WCHAR thực ra là UTF-16LE.
                # Nếu vẫn lỗi với VARCHAR thường (SQL_CHAR), thêm:
                #   conn.setdecoding(pyodbc.SQL_CHAR, encoding='cp1252')
                pass

                self._local.conn = conn
                logger.debug(
                    f"DB connection OK — thread: {threading.current_thread().name}"
                )
            except pyodbc.Error as e:
                logger.error(f"Không thể kết nối SQL Server: {e}")
                
                # Cung cấp gợi ý sửa lỗi cụ thể hơn
                error_msg = (
                    f"Lỗi kết nối SQL Server (Server: {db_config.server}, Database: {db_config.database}).\n"
                    f"Chi tiết kỹ thuật: {e}\n\n"
                    f"Hướng dẫn khắc phục:\n"
                    f"  1. Đảm bảo dịch vụ SQL Server (MSSQLSERVER hoặc SQLEXPRESS) đang chạy.\n"
                    f"  2. Kiểm tra chuỗi kết nối trong config.py hoặc biến môi trường DB_SERVER.\n"
                    f"  3. Thử đổi server thành '.' hoặc '(local)' nếu đang dùng máy cá nhân.\n"
                    f"  4. Đảm bảo Database '{db_config.database}' đã được khởi tạo."
                )
                raise ConnectionError(error_msg)
        return self._local.conn

    # ─── Public API ────────────────────────────

    @contextmanager
    def get_cursor(self, commit: bool = False):
        """
        Context manager trả về cursor.
        Tự động commit/rollback và đóng cursor.

            with db.get_cursor() as cur:
                cur.execute("SELECT ...")
                rows = cur.fetchall()

            with db.get_cursor(commit=True) as cur:
                cur.execute("INSERT ...")
        """
        conn   = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            if commit:
                conn.commit()
        except pyodbc.Error as e:
            try:
                conn.rollback()
            except Exception:
                pass
            logger.error(f"DB Error: {e}")
            raise
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            try:
                cursor.close()
            except Exception:
                pass

    def get_connection(self) -> pyodbc.Connection:
        """Lấy raw connection (dùng cho các thao tác đặc biệt)."""
        return self._get_connection()

    def execute(self, sql: str, params=None, commit: bool = False) -> list:
        """
        Chạy 1 câu SQL, trả về list of rows.

        params:
          None hoặc ()  → không truyền params
          (val,)        → 1 param
          (v1, v2, ...) → nhiều params

        QUAN TRỌNG: params=() là falsy nên sẽ gọi execute(sql) không có params.
        Nếu SQL thật sự cần truyền () (không param) → dùng params=None.
        """
        with self.get_cursor(commit=commit) as cur:
            if params:                      # None, (), [] đều bỏ qua
                cur.execute(sql, params)
            else:
                cur.execute(sql)
            try:
                return list(cur.fetchall())
            except pyodbc.ProgrammingError:
                # INSERT/UPDATE/DELETE không có fetchall
                return []

    def execute_many(self, sql: str, params_list: list, commit: bool = True) -> int:
        """Bulk insert/update."""
        with self.get_cursor(commit=commit) as cur:
            cur.executemany(sql, params_list)
            return cur.rowcount

    def call_procedure(self, proc_name: str, params: tuple = ()) -> list:
        """Gọi Stored Procedure."""
        if params:
            placeholders = ", ".join(["?"] * len(params))
            sql = f"EXEC {proc_name} {placeholders}"
            return self.execute(sql, params)
        else:
            return self.execute(f"EXEC {proc_name}")

    def test_connection(self) -> bool:
        """Kiểm tra kết nối còn sống không."""
        try:
            rows = self.execute("SELECT 1")
            return bool(rows)
        except Exception as e:
            logger.warning(f"Connection test failed: {e}")
            self.reset_connection()
            return False

    def reset_connection(self):
        """Đóng và xoá connection của thread hiện tại để tạo mới lần sau."""
        if hasattr(self._local, "conn") and self._local.conn:
            try:
                self._local.conn.close()
            except Exception:
                pass
        self._local.conn = None

    def close(self):
        """Alias của reset_connection."""
        self.reset_connection()


# ─────────────────────────────────────────────
#  Singleton — dùng trong toàn project
# ─────────────────────────────────────────────
db = DatabaseConnection()


def get_db() -> DatabaseConnection:
    return db


# ─── Quick test ──────────────────────────────
if __name__ == "__main__":
    logger.info("Test kết nối SQL Server...")
    try:
        if db.test_connection():
            logger.success("✅ Kết nối thành công!")
            rows = db.execute("SELECT name FROM sys.databases ORDER BY name")
            logger.info(f"Databases: {[r[0] for r in rows]}")

            rows = db.execute("SELECT COUNT(*) FROM Students")
            logger.info(f"Students count: {rows[0][0]}")
        else:
            logger.error("❌ Kết nối thất bại")
    except ConnectionError as e:
        logger.error(str(e))