import json
import time
import threading
from datetime import datetime
from core.logger import logger, attendance_logger
import redis
from pydantic import BaseModel

import sys
from pathlib import Path
# Đảm bảo import được các module của Server
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.config import redis_config
from database.repositories import record_repo, session_repo

REDIS_QUEUE_KEY = "queue:attendance"
REDIS_DLQ_KEY = "queue:attendance_dlq"
MAX_RETRIES = 3

class AttendanceTask(BaseModel):
    session_id: int
    student_id: int
    student_code: str
    full_name: str
    class_name: str
    recognition_score: float
    camera_id: int
    timestamp: str
    retry_count: int = 0

class AttendanceWorker:
    """
    Worker xử lý điểm danh bất đồng bộ qua Redis Queue.
    Giảm tải cho FastAPI, chống nghẽn MySQL khi có hàng trăm camera.
    """
    def __init__(self):
        self.redis = redis.Redis.from_url(redis_config.url, decode_responses=True)
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="Attendance-Worker")

    def start(self):
        try:
            self.redis.ping()
            logger.info("🚀 Attendance Worker khởi động - Sẵn sàng xử lý Queue.")
            self.thread.start()
        except Exception as e:
            logger.error(f"❌ Worker không thể kết nối Redis: {e}")

    def stop(self):
        logger.info("🛑 Đang dừng Attendance Worker (Graceful Shutdown)...")
        self._stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        logger.info("💤 Attendance Worker đã dừng an toàn.")

    def _run(self):
        while not self._stop_event.is_set():
            try:
                # Lấy task từ queue, block tối đa 2s để có cơ hội check _stop_event
                result = self.redis.brpop(REDIS_QUEUE_KEY, timeout=2)
                if result:
                    _, task_json = result
                    self._process_task(task_json)
                else:
                    # Rảnh rỗi, có thể log queue size nếu cần
                    pass
            except redis.ConnectionError:
                logger.warning("Worker mất kết nối Redis, đang thử lại...")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Worker Error: {e}")
                time.sleep(1)

    def _process_task(self, task_json: str):
        try:
            task_dict = json.loads(task_json)
            task = AttendanceTask(**task_dict)
        except Exception as e:
            logger.error(f"❌ Invalid task format: {task_json} | {e}")
            self._send_to_dlq(task_json, "Invalid Format")
            return

        logger.debug(f"⚙️ Đang xử lý điểm danh: {task.full_name} ({task.student_code})")

        try:
            # 1. Validate Session Active (Chặn nếu session bị đóng)
            session = session_repo.get_by_id(task.session_id)
            if not session or session.status != "ACTIVE":
                logger.warning(f"⚠️ Từ chối điểm danh - Session {task.session_id} không hợp lệ hoặc đã đóng.")
                return

            # 2. Validate Duplicate (Đã điểm danh chưa?)
            already = record_repo.is_already_recorded(task.session_id, task.student_id)
            if already:
                logger.warning(f"⏩ [SKIP] {task.full_name} đã điểm danh trong Session {task.session_id}.")
                return

            # 3. Insert DB
            success = record_repo.record_attendance(
                session_id=task.session_id,
                student_id=task.student_id,
                recognition_score=task.recognition_score,
                camera_id=task.camera_id
            )

            if success:
                attendance_logger.info(f"✅ Ghi DB thành công: {task.full_name} (Session: {task.session_id} - Score: {task.recognition_score:.2f})")
                
                # Gửi thông báo qua Telegram (optional)
                try:
                    import sys
                    from pathlib import Path
                    root_dir = Path(__file__).parent.parent.parent
                    if str(root_dir) not in sys.path:
                        sys.path.insert(0, str(root_dir))
                    from telegram_notifier import send_telegram_msg
                    
                    # 1. Tuỳ chọn: Gửi báo danh từng người (có thể bị nhiều tin nhắn, có thể comment lại)
                    # msg = f"✅ ĐIỂM DANH (WORKER)\n👤 Học viên: {task.full_name}\n🆔 MSSV: {task.student_code}\n🏫 Lớp: {task.class_name}\n🕒 Thời gian: {task.timestamp}\n🎯 Độ tin cậy: {task.recognition_score*100:.1f}%"
                    # threading.Thread(target=send_telegram_msg, args=(msg,), daemon=True).start()
                    
                    # 2. Kiểm tra nếu lớp ĐỦ thì gửi thông báo
                    absent_count = record_repo.get_class_absent_count(task.session_id, task.class_name)
                    if absent_count == 0:
                        redis_key = f"notified_full_{task.session_id}_{task.class_name}"
                        if not self.redis.get(redis_key):
                            self.redis.set(redis_key, "1", ex=86400) # Lưu 1 ngày
                            msg_full = f"{task.class_name} - Đủ"
                            threading.Thread(target=send_telegram_msg, args=(msg_full,), daemon=True).start()
                            
                except Exception as e:
                    logger.error(f"Lỗi gửi Telegram (Lớp đủ): {e}")
            else:
                raise Exception("Hàm record_attendance trả về False")

        except Exception as e:
            logger.error(f"❌ Lỗi ghi DB cho {task.full_name}: {e}")
            self._handle_failure(task)

    def _handle_failure(self, task: AttendanceTask):
        task.retry_count += 1
        if task.retry_count <= MAX_RETRIES:
            logger.info(f"🔄 Re-queueing task {task.student_code} (Lần {task.retry_count}/{MAX_RETRIES})")
            # Đẩy lại vào queue (dùng lpush để xử lý sớm hoặc rpush để xử lý sau)
            self.redis.lpush(REDIS_QUEUE_KEY, task.model_dump_json())
        else:
            logger.error(f"💀 Dead-letter (DLQ) cho {task.student_code} sau {MAX_RETRIES} lần thử.")
            self._send_to_dlq(task.model_dump_json(), "Max Retries Exceeded")

    def _send_to_dlq(self, payload_str: str, reason: str):
        try:
            dlq_item = {
                "payload": payload_str,
                "reason": reason,
                "failed_at": datetime.now().isoformat()
            }
            self.redis.lpush(REDIS_DLQ_KEY, json.dumps(dlq_item))
        except Exception as e:
            logger.error(f"Không thể ghi vào DLQ: {e}")

# Khởi tạo instance toàn cục cho FastAPI
attendance_worker = AttendanceWorker()

if __name__ == "__main__":
    attendance_worker.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        attendance_worker.stop()
