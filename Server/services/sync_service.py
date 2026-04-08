import sqlite3
import threading
import time
from datetime import datetime
from loguru import logger
from pathlib import Path
import sys

# Ensure root in path for relative imports
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database.repositories import record_repo
from config import BASE_DIR

class SyncService:
    """
    Dịch vụ đồng bộ dữ liệu (Edge Processing).
    Lưu trữ tạm thời vào SQLite khi mất kết nối hoặc để giảm độ trễ,
    sau đó tự động đẩy lên SQL Server khi có mạng.
    """
    def __init__(self):
        self.db_path = BASE_DIR / "database" / "local_buffer.db"
        self._init_db()
        self._stop_event = threading.Event()
        self._sync_thread = threading.Thread(target=self._sync_loop, name="Sync-Thread", daemon=True)
        self._sync_thread.start()

    def _init_db(self):
        """Khởi tạo cấu trúc bảng SQLite cục bộ."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS offline_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER,
                        student_id INTEGER,
                        check_in_time TEXT,
                        recognition_score REAL,
                        snapshot_path TEXT,
                        camera_id INTEGER,
                        synced INTEGER DEFAULT 0
                    )
                """)
            logger.info(f"Đã khởi tạo DB local buffer: {self.db_path}")
        except Exception as e:
            logger.error(f"Lỗi khởi tạo DB local: {e}")

    def save_offline(self, session_id, student_id, check_in_time, score, snapshot_path, camera_id):
        """Lưu bản ghi vào DB cục bộ ngay lập tức."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO offline_records 
                    (session_id, student_id, check_in_time, recognition_score, snapshot_path, camera_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id, 
                    student_id, 
                    check_in_time.isoformat() if hasattr(check_in_time, "isoformat") else str(check_in_time), 
                    score, 
                    snapshot_path, 
                    camera_id
                ))
            logger.info(f"➜ [OFFLINE] Đã lưu đệm student_id={student_id} vào SQLite")
            return True
        except Exception as e:
            logger.error(f"Lỗi lưu offline record: {e}")
            return False

    def _sync_loop(self):
        """Vòng lặp chạy ngầm kiểm tra và đồng bộ."""
        # Chờ hệ thống khởi động ổn định
        time.sleep(5)
        while not self._stop_event.is_set():
            try:
                self._do_sync()
            except Exception as e:
                logger.debug(f"Chu kỳ sync tạm dừng (mất mạng?): {e}")
            
            # Kiểm tra mỗi 30 giây
            for _ in range(30):
                if self._stop_event.is_set(): break
                time.sleep(1)

    def _do_sync(self):
        """Thực hiện đẩy dữ liệu từ SQLite lên SQL Server."""
        with sqlite3.connect(self.db_path) as conn:
            # Lấy các bản ghi chưa đồng bộ
            rows = conn.execute(
                "SELECT id, session_id, student_id, check_in_time, recognition_score, snapshot_path, camera_id "
                "FROM offline_records WHERE synced = 0"
            ).fetchall()

            if not rows:
                return

            logger.info(f"🔄 Đang đồng bộ {len(rows)} bản ghi lên Server...")
            
            for row in rows:
                rid, sid, stid, ts_str, score, path, cam = row
                
                try:
                    # Chuyển string sang datetime
                    ts = datetime.fromisoformat(ts_str)
                    
                    # Gọi repo để lưu vào SQL Server
                    # Dùng upsert để giữ đúng timestamp nguyên bản
                    success = record_repo.upsert(
                        session_id=sid,
                        student_id=stid,
                        status='PRESENT',
                        check_in_time=ts,
                        recognition_score=score
                    )
                    
                    if success:
                        conn.execute("UPDATE offline_records SET synced = 1 WHERE id = ?", (rid,))
                        conn.commit()
                        logger.success(f"✅ Đồng bộ thành công student_id={stid}")
                    else:
                        # Nếu fail (Vẫn mất mạng), dừng batch này lại để đợi chu kỳ sau
                        logger.warning("Đồng bộ thất bại, có thể Server vẫn offline.")
                        break
                except Exception as e:
                    logger.error(f"Lỗi khi đồng bộ dòng {rid}: {e}")
                    break

    def stop(self):
        self._stop_event.set()
        if self._sync_thread.is_alive():
            self._sync_thread.join(timeout=2)

# Singleton
sync_service = SyncService()
