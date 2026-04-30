"""
services/enrollment_service.py
Đăng ký khuôn mặt học viên.
"""
import time
import threading
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field

import cv2
import numpy as np
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ai_config, app_config
from database.repositories import student_repo, embedding_repo, class_repo
from services.face_engine import face_engine
from services.embedding_cache_manager import cache_manager


# ─────────────────────────────────────────────
@dataclass
class EnrollmentResult:
    success:       bool
    student_id:    int
    student_code:  str
    full_name:     str
    photos_taken:  int
    photos_valid:  int
    avg_det_score: float
    error_msg:     str = ""

    @property
    def summary(self) -> str:
        if self.success:
            return (
                f"✅ Đăng ký thành công!\n"
                f"   Học viên: [{self.student_code}] {self.full_name}\n"
                f"   Ảnh hợp lệ: {self.photos_valid}/{self.photos_taken}\n"
                f"   Chất lượng: {self.avg_det_score*100:.1f}%"
            )
        return f"❌ Đăng ký thất bại: {self.error_msg}"


@dataclass
class CaptureSession:
    target_count:  int = 10
    frames:        list = field(default_factory=list)
    is_capturing:  bool = False
    is_done:       bool = False

    @property
    def count(self) -> int:
        return len(self.frames)

    @property
    def progress(self) -> float:
        return self.count / self.target_count

    @property
    def is_complete(self) -> bool:
        return self.count >= self.target_count


# ─────────────────────────────────────────────
class EnrollmentService:

    def __init__(self):
        self._capture:    Optional[CaptureSession] = None
        self._student_id: Optional[int] = None
        self._lock        = threading.Lock()

        self.on_progress:     Optional[Callable[[int, int], None]] = None
        self.on_face_detected: Optional[Callable[[bool], None]] = None

    # ─── Quản lý học viên ─────────────────────

    def get_all_classes(self):
        return class_repo.get_all()

    def create_student(
        self,
        student_code: str,
        full_name:    str,
        class_id:     int  = None,
        gender:       str  = None,
        phone:        str  = None,
        email:        str  = None,
        class_name:   str  = None,
        building:     str  = None,
        floor:        str  = None,
        room:         str  = None,
    ) -> Optional[int]:
        """Tạo học viên mới. Trả về student_id hoặc None nếu lỗi."""
        existing = student_repo.get_by_code(student_code)
        if existing:
            logger.warning(f"Mã học viên [{student_code}] đã tồn tại!")
            return None

        student_id = student_repo.create(
            student_code=student_code,
            full_name=full_name,
            class_id=class_id,
            gender=gender,
            phone=phone,
            email=email,
            class_name=class_name,
            building=building,
            floor=floor,
            room=room,
        )
        logger.info(f"Tạo học viên: [{student_code}] {full_name} (id={student_id})")
        return student_id

    def get_students(self, class_id: int = None):
        return student_repo.get_all(class_id=class_id)

    def search_students(self, keyword: str):
        return student_repo.search(keyword)

    # ─── Capture ảnh ──────────────────────────

    def start_capture(self, student_id: int, photo_count: int = None):
        count = photo_count or ai_config.max_enroll_photos
        with self._lock:
            self._student_id = student_id
            self._capture    = CaptureSession(target_count=count)
            self._capture.is_capturing = True
        logger.info(f"Bắt đầu chụp ảnh: student_id={student_id}, target={count} ảnh")

    def add_frame(self, frame: np.ndarray) -> bool:
        with self._lock:
            if not self._capture or not self._capture.is_capturing:
                return False
            if self._capture.is_complete:
                return False

        # Sử dụng OpenCV siêu nhẹ thay vì model nặng
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces_rect = face_cascade.detectMultiScale(gray, 1.3, 5)
        has_face = len(faces_rect) > 0

        if self.on_face_detected:
            try:
                self.on_face_detected(has_face)
            except Exception:
                pass

        if not has_face:
            return False

        with self._lock:
            if not self._capture.is_complete:
                self._capture.frames.append(frame.copy())
                count = self._capture.count
                total = self._capture.target_count

        if self.on_progress:
            try:
                self.on_progress(count, total)
            except Exception:
                pass

        logger.debug(f"Ảnh {count}/{total} chụp OK")
        return True

    def add_frame_with_delay(self, frame: np.ndarray, min_gap_sec: float = 0.3) -> bool:
        with self._lock:
            if not self._capture:
                return False
            last_time = getattr(self._capture, '_last_capture_time', 0)
            if time.time() - last_time < min_gap_sec:
                return False
            self._capture._last_capture_time = time.time()
        return self.add_frame(frame)

    def cancel_capture(self):
        with self._lock:
            self._capture    = None
            self._student_id = None
        logger.info("Huỷ capture enrollment")

    @property
    def capture_progress(self) -> tuple[int, int]:
        with self._lock:
            if not self._capture:
                return 0, 0
            return self._capture.count, self._capture.target_count

    @property
    def is_capture_complete(self) -> bool:
        with self._lock:
            return bool(self._capture and self._capture.is_complete)

    # ─── Tính embedding và lưu DB ─────────────

    def finish_enrollment(self) -> EnrollmentResult:
        with self._lock:
            if not self._capture or not self._student_id:
                return EnrollmentResult(
                    success=False, student_id=0,
                    student_code="", full_name="",
                    photos_taken=0, photos_valid=0,
                    avg_det_score=0, error_msg="Chưa bắt đầu chụp ảnh"
                )
            frames     = self._capture.frames.copy()
            student_id = self._student_id

        student = student_repo.get_by_id(student_id)
        if not student:
            return EnrollmentResult(
                success=False, student_id=student_id,
                student_code="", full_name="",
                photos_taken=len(frames), photos_valid=0,
                avg_det_score=0,
                error_msg=f"Không tìm thấy học viên id={student_id}"
            )

        if len(frames) < ai_config.min_enroll_photos:
            return EnrollmentResult(
                success=False,
                student_id=student_id,
                student_code=student.student_code,
                full_name=student.full_name,
                photos_taken=len(frames),
                photos_valid=0,
                avg_det_score=0,
                error_msg=(
                    f"Cần tối thiểu {ai_config.min_enroll_photos} ảnh, "
                    f"mới chụp được {len(frames)} ảnh"
                )
            )

        # Background Task tự động trích xuất Vector
        def _extract_bg():
            try:
                logger.info(f"⚙️ [Background] Tính embedding cho [{student.student_code}] từ {len(frames)} ảnh...")
                embedding, avg_score, valid_count = face_engine.compute_enrollment_embedding(frames)
                if embedding is not None and valid_count >= ai_config.min_enroll_photos:
                    embedding_repo.save_embedding(
                        student_id=student_id,
                        embedding=embedding,
                        model_version="buffalo_l",
                    )
                    student_repo.update_enrollment_status(student_id, enrolled=True)
                    self._save_profile_photo(frames, student.student_code)
                    
                    cache_manager.add_student_to_cache(
                        student_id=student_id,
                        student_code=student.student_code,
                        full_name=student.full_name,
                        class_id=student.class_id or 0,
                        class_name=student.class_name or "",
                        embedding=embedding,
                    )
                    logger.success(f"✅ [Background] Cập nhật AI xong cho: {student.full_name}")
                else:
                    logger.error(f"❌ [Background] Nhận diện thất bại hoặc không đủ ảnh hợp lệ cho {student.full_name}")
            except Exception as e:
                logger.error(f"❌ [Background] Lỗi trích xuất Vector: {e}")

        threading.Thread(target=_extract_bg, daemon=True).start()

        with self._lock:
            self._capture    = None
            self._student_id = None

        result = EnrollmentResult(
            success=True,
            student_id=student_id,
            student_code=student.student_code,
            full_name=student.full_name,
            photos_taken=len(frames),
            photos_valid=len(frames),
            avg_det_score=1.0,
        )
        logger.success(result.summary)
        return result

    def reenroll_student(self, student_id: int, frames: list[np.ndarray]) -> EnrollmentResult:
        self.start_capture(student_id, photo_count=len(frames))
        with self._lock:
            if self._capture:
                self._capture.frames = frames
        return self.finish_enrollment()

    def _save_profile_photo(self, frames: list[np.ndarray], student_code: str):
        try:
            photo_dir = app_config.snapshot_dir.parent / "profiles"
            photo_dir.mkdir(exist_ok=True)
            path = photo_dir / f"{student_code}_profile.jpg"
            if frames:
                cv2.imwrite(str(path), frames[0])
                logger.debug(f"Profile photo saved: {path}")
        except Exception as e:
            logger.warning(f"Không lưu được profile photo: {e}")


# ─────────────────────────────────────────────
enrollment_service = EnrollmentService()