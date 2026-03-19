"""
services/attendance_service.py
Logic nghiệp vụ điểm danh.
"""
import time
import threading
import queue
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ai_config, app_config
from database.repositories import session_repo, record_repo
from database.models import AttendanceSession
from services.face_engine import RecognitionResult


# ─────────────────────────────────────────────
@dataclass
class AttendanceEvent:
    student_id:    int
    student_code:  str
    full_name:     str
    class_id:      int
    check_in_time: datetime
    similarity:    float
    snapshot_path: Optional[str]
    session_id:    int

    @property
    def time_str(self) -> str:
        return self.check_in_time.strftime("%H:%M:%S")

    @property
    def score_pct(self) -> str:
        return f"{self.similarity * 100:.1f}%"


# ─────────────────────────────────────────────
class AttendanceService:

    def __init__(self):
        self._session_id:   Optional[int] = None
        self._session:      Optional[AttendanceSession] = None
        self._active:       bool = False

        self._cooldown_map: dict[int, float] = {}
        self._cooldown_lock = threading.Lock()

        self.on_attendance: Optional[Callable[[AttendanceEvent], None]] = None
        self.on_duplicate:  Optional[Callable[[str, float], None]] = None

        self._stats = {
            "total_recognized": 0,
            "total_recorded":   0,
            "total_duplicate":  0,
            "session_start":    None,
        }

        # NÂNG CẤP: Khởi tạo luồng chạy ngầm để ghi Database
        self._db_queue = queue.Queue()
        self._stop_worker = threading.Event()
        self._worker_thread = threading.Thread(target=self._db_worker, name="DB-Worker", daemon=True)
        self._worker_thread.start()

    def _db_worker(self):
        """Luồng chạy ngầm: Lấy dữ liệu từ hàng đợi và ghi vào SQL Server / Ổ cứng"""
        while not self._stop_worker.is_set():
            try:
                task = self._db_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            event, frame_crop, camera_id = task
            snapshot_path = None
            
            # 1. Lưu ảnh ra ổ cứng (Async)
            if app_config.save_snapshots and frame_crop is not None:
                try:
                    ts = event.check_in_time.strftime("%H%M%S")
                    fname = f"{event.student_code}_{ts}_cam{camera_id}.jpg"
                    path = app_config.snapshot_dir / fname
                    cv2.imwrite(str(path), frame_crop)
                    snapshot_path = str(path)
                    event.snapshot_path = snapshot_path
                except Exception as e:
                    logger.error(f"Lỗi lưu snapshot ngầm: {e}")

            # 2. Ghi vào SQL Server (Async)
            success = record_repo.record_attendance(
                session_id=event.session_id,
                student_id=event.student_id,
                recognition_score=event.similarity,
                snapshot_path=snapshot_path,
                camera_id=camera_id,
            )

            if success:
                self._stats["total_recorded"] += 1
            else:
                logger.error(f"Ghi DB thất bại cho student_id={event.student_id}")

            self._db_queue.task_done()

    # ─── Quản lý Session ──────────────────────

    def create_session(
        self,
        class_id:     int,
        subject_name: str,
        session_date: date = None,
    ) -> int:
        """Tạo buổi điểm danh mới, tạo sẵn records ABSENT cho toàn bộ lớp."""
        sid = session_repo.create_session(
            class_id=class_id,
            subject_name=subject_name,
            session_date=session_date,
        )
        logger.info(f"Tạo session: id={sid}, class={class_id}, subject={subject_name}")
        return sid

    def start_session(self, session_id: int) -> bool:
        """PENDING → ACTIVE."""
        if self._active:
            logger.warning("Đang có buổi điểm danh, kết thúc buổi cũ trước!")
            return False

        self._session = session_repo.get_by_id(session_id)
        if not self._session:
            logger.error(f"Không tìm thấy session {session_id}")
            return False

        session_repo.start_session(session_id)
        self._session_id = session_id
        self._active     = True

        with self._cooldown_lock:
            self._cooldown_map.clear()
        self._reset_stats()

        logger.success(
            f"▶ Bắt đầu điểm danh: [{self._session.subject_name}] "
            f"| Lớp: {self._session.class_name} "
            f"| Session ID: {session_id}"
        )
        return True

    def end_session(self) -> Optional[AttendanceSession]:
        """ACTIVE → COMPLETED. Trả về thông tin tổng kết."""
        if not self._active or not self._session_id:
            logger.warning("Không có buổi điểm danh nào đang chạy")
            return None

        # Đợi các task ghi DB chạy ngầm hoàn tất trước khi đóng session
        logger.info("Đang chờ các bản ghi DB cuối cùng lưu xong...")
        self._db_queue.join()

        session_repo.end_session(self._session_id)
        session = session_repo.get_by_id(self._session_id)
        self._active     = False
        self._session_id = None
        self._session    = None

        logger.success(
            f"⏹ Kết thúc điểm danh | "
            f"Có mặt: {session.present_count} | "
            f"Vắng: {session.absent_count}"
        )
        return session

    # ─── Xử lý kết quả nhận diện ──────────────

    def process_recognition(
        self,
        result:    RecognitionResult,
        frame:     Optional[np.ndarray] = None,
        camera_id: int = 1,
    ) -> Optional[AttendanceEvent]:
        if not self._active:
            return None
        if not result.recognized:
            return None
            
        # NÂNG CẤP: Chống điểm danh hộ qua ảnh/video
        if not result.is_real:
            logger.warning(f"Từ chối điểm danh do phát hiện SPOOF: {result.full_name or 'Unknown'}")
            return None

        self._stats["total_recognized"] += 1
        student_id = result.student_id

        # Kiểm tra cooldown
        remaining = self._get_cooldown_remaining(student_id)
        if remaining > 0:
            self._stats["total_duplicate"] += 1
            logger.debug(
                f"COOLDOWN [{result.full_name}] — còn {remaining:.0f}s"
            )
            if self.on_duplicate:
                try:
                    self.on_duplicate(result.full_name, remaining)
                except Exception:
                    pass
            return None

        self._set_cooldown(student_id)

        event = AttendanceEvent(
            student_id=student_id,
            student_code=result.student_code,
            full_name=result.full_name,
            class_id=result.class_id,
            check_in_time=datetime.now(),
            similarity=result.similarity,
            snapshot_path=None,
            session_id=self._session_id,
        )

        # Cắt ảnh trên RAM (việc ghi đĩa sẽ do _db_worker thực hiện)
        face_crop = None
        if app_config.save_snapshots and frame is not None:
            try:
                x1, y1, x2, y2 = result.bbox
                pad = 20
                h, w = frame.shape[:2]
                x1 = max(0, x1 - pad);  y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad);  y2 = min(h, y2 + pad)
                face_crop = frame[y1:y2, x1:x2].copy()
            except Exception as e:
                logger.error(f"Lỗi khi cắt ảnh: {e}")

        # Đẩy vào queue để luồng phụ lưu DB và ảnh
        self._db_queue.put((event, face_crop, camera_id))

        logger.success(
            f"✅ ĐIỂM DANH: [{result.student_code}] {result.full_name} "
            f"| Score: {result.similarity:.3f} | {event.time_str}"
        )

        if self.on_attendance:
            try:
                self.on_attendance(event)
            except Exception as e:
                logger.error(f"on_attendance callback error: {e}")

        return event

    def process_frame_results(
        self,
        results:   list[RecognitionResult],
        frame:     Optional[np.ndarray] = None,
        camera_id: int = 1,
    ) -> list[AttendanceEvent]:
        events = []
        for result in results:
            event = self.process_recognition(result, frame, camera_id)
            if event:
                events.append(event)
        return events

    # ─── Cooldown ─────────────────────────────

    def _get_cooldown_remaining(self, student_id: int) -> float:
        with self._cooldown_lock:
            last_time = self._cooldown_map.get(student_id, 0)
            elapsed   = time.time() - last_time
            return max(0.0, ai_config.attendance_cooldown_sec - elapsed)

    def _set_cooldown(self, student_id: int):
        with self._cooldown_lock:
            self._cooldown_map[student_id] = time.time()

    def reset_cooldown(self, student_id: int = None):
        with self._cooldown_lock:
            if student_id:
                self._cooldown_map.pop(student_id, None)
            else:
                self._cooldown_map.clear()
        logger.info(
            f"Reset cooldown: {'tất cả' if not student_id else f'student {student_id}'}"
        )

    # ─── Truy vấn trạng thái ──────────────────

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def session_id(self) -> Optional[int]:
        return self._session_id

    @property
    def current_session(self) -> Optional[AttendanceSession]:
        return self._session

    def get_present_list(self) -> list[dict]:
        if not self._session_id:
            return []
        return record_repo.get_present_list(self._session_id)

    def get_stats(self) -> dict:
        stats = self._stats.copy()
        if stats["session_start"]:
            elapsed = (datetime.now() - stats["session_start"]).seconds
            stats["elapsed_sec"] = elapsed
            stats["elapsed_str"] = f"{elapsed//60:02d}:{elapsed%60:02d}"
        stats["session_id"]  = self._session_id
        stats["is_active"]   = self._active
        with self._cooldown_lock:
            stats["in_cooldown"] = len(self._cooldown_map)
        return stats

    def _reset_stats(self):
        self._stats = {
            "total_recognized": 0,
            "total_recorded":   0,
            "total_duplicate":  0,
            "session_start":    datetime.now(),
        }


# ─────────────────────────────────────────────
attendance_service = AttendanceService()