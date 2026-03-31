#database/repositories.py
import numpy as np
from datetime import datetime, date
from typing import Optional
from loguru import logger

from .connection import get_db
from .models import (
    Class, Student, FaceEmbedding, AttendanceSession,
    AttendanceRecord, Camera, EmbeddingCache,
)


# ══════════════════════════════════════════════
#  CLASS REPOSITORY
# ══════════════════════════════════════════════
class ClassRepository:
    _SQL = """
        SELECT class_id, class_code, class_name, teacher_name,
               academic_year, is_active, created_at
        FROM Classes
    """

    def get_all(self, active_only: bool = True) -> list:
        if active_only:
            rows = get_db().execute(f"{self._SQL} WHERE is_active = 1 ORDER BY class_name")
        else:
            rows = get_db().execute(f"{self._SQL} ORDER BY class_name")
        return [Class(*r) for r in rows]

    def get_by_id(self, class_id: int) -> Optional[Class]:
        rows = get_db().execute(f"{self._SQL} WHERE class_id = ?", (class_id,))
        return Class(*rows[0]) if rows else None

    def create(self, class_code: str, class_name: str,
               teacher_name: str = None, academic_year: str = None) -> int:
        rows = get_db().execute(
            """
            INSERT INTO Classes (class_code, class_name, teacher_name, academic_year)
            OUTPUT INSERTED.class_id VALUES (?, ?, ?, ?)
            """,
            (class_code, class_name, teacher_name, academic_year),
            commit=True,
        )
        cid = rows[0][0] if rows else -1
        logger.info(f"Created class [{class_code}] id={cid}")
        return cid

    def update(self, class_id: int, **kwargs) -> bool:
        allowed = {"class_name", "teacher_name", "academic_year", "is_active"}
        cols = {k: v for k, v in kwargs.items() if k in allowed}
        if not cols:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in cols)
        params = list(cols.values()) + [class_id]
        get_db().execute(
            f"UPDATE Classes SET {set_clause} WHERE class_id = ?",
            tuple(params), commit=True,
        )
        return True


# ══════════════════════════════════════════════
#  STUDENT REPOSITORY
# ══════════════════════════════════════════════
class StudentRepository:
    # 11 cols: 10 từ bảng + class_name từ JOIN
    # Thứ tự PHẢI khớp với Student dataclass
    _SQL = """
        SELECT s.student_id, s.student_code, s.full_name,
               s.gender, s.date_of_birth,
               s.phone, s.email, s.class_id,
               s.face_enrolled, s.created_at,
               c.class_name
        FROM Students s
        LEFT JOIN Classes c ON c.class_id = s.class_id
    """

    def get_all(self, class_id: int = None) -> list:
        where_parts = []
        params = []
        if class_id is not None:
            where_parts.append("s.class_id = ?")
            params.append(class_id)
        where = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
        sql   = f"{self._SQL} {where} ORDER BY s.full_name"
        rows  = get_db().execute(sql, tuple(params) if params else None)
        return [Student(*r) for r in rows]

    def get_by_id(self, student_id: int) -> Optional[Student]:
        rows = get_db().execute(f"{self._SQL} WHERE s.student_id = ?", (student_id,))
        return Student(*rows[0]) if rows else None

    def get_by_code(self, student_code: str) -> Optional[Student]:
        rows = get_db().execute(f"{self._SQL} WHERE s.student_code = ?", (student_code,))
        return Student(*rows[0]) if rows else None

    def create(self, student_code: str, full_name: str,
               class_id: int = None, gender: str = None,
               phone: str = None, email: str = None) -> int:
        rows = get_db().execute(
            """
            INSERT INTO Students (student_code, full_name, gender, class_id, phone, email)
            OUTPUT INSERTED.student_id VALUES (?, ?, ?, ?, ?, ?)
            """,
            (student_code, full_name, gender, class_id, phone, email),
            commit=True,
        )
        sid = rows[0][0] if rows else -1
        logger.info(f"Created student [{student_code}] {full_name} id={sid}")
        return sid

    def update(self, student_id: int, **kwargs) -> bool:
        allowed = {"student_code", "full_name", "gender", "date_of_birth",
                   "phone", "email", "class_id"}
        cols = {k: v for k, v in kwargs.items() if k in allowed}
        if not cols:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in cols)
        params = list(cols.values()) + [student_id]
        get_db().execute(
            f"UPDATE Students SET {set_clause} WHERE student_id = ?",
            tuple(params), commit=True,
        )
        return True

    def update_enrollment_status(self, student_id: int, enrolled: bool = True) -> bool:
        get_db().execute(
            "UPDATE Students SET face_enrolled = ? WHERE student_id = ?",
            (1 if enrolled else 0, student_id),
            commit=True,
        )
        logger.info(f"Student {student_id} face_enrolled → {enrolled}")
        return True

    def search(self, keyword: str) -> list:
        kw  = f"%{keyword}%"
        sql = f"""
            {self._SQL}
            WHERE s.full_name LIKE ? OR s.student_code LIKE ?
            ORDER BY s.full_name
        """
        rows = get_db().execute(sql, (kw, kw))
        return [Student(*r) for r in rows]

    def get_count_by_class(self, class_id: int) -> int:
        rows = get_db().execute(
            "SELECT COUNT(*) FROM Students WHERE class_id = ?", (class_id,)
        )
        return rows[0][0] if rows else 0

    def delete(self, student_id: int) -> bool:
        try:
            db = get_db()
            # Xóa lịch sử điểm danh trước để tránh lỗi Foreign Key
            db.execute("DELETE FROM AttendanceRecords WHERE student_id = ?", (student_id,), commit=True)
            # Xóa dữ liệu khuôn mặt liên quan
            db.execute("DELETE FROM FaceEmbeddings WHERE student_id = ?", (student_id,), commit=True)
            # Cuối cùng xóa học viên
            db.execute(
                "DELETE FROM Students WHERE student_id = ?",
                (student_id,), commit=True,
            )
            return True
        except Exception as e:
            logger.error(f"Cannot delete student {student_id}: {e}")
            return False


# ══════════════════════════════════════════════
#  FACE EMBEDDING REPOSITORY
# ══════════════════════════════════════════════
class FaceEmbeddingRepository:

    def save_embedding(self, student_id: int, embedding: np.ndarray,
                       model_version: str = "buffalo_l") -> bool:
        """Deactivate cũ → insert mới (đảm bảo unique index)."""
        get_db().execute(
            "UPDATE FaceEmbeddings SET is_active = 0 WHERE student_id = ?",
            (student_id,), commit=True,
        )
        get_db().execute(
            """
            INSERT INTO FaceEmbeddings (student_id, embedding_vector, model_version)
            VALUES (?, ?, ?)
            """,
            (student_id, embedding.astype(np.float32).tobytes(), model_version),
            commit=True,
        )
        logger.info(f"Saved embedding: student={student_id}")
        return True

    def save(self, student_id: int, embedding: np.ndarray,
             model_version: str = "buffalo_l") -> int:
        self.save_embedding(student_id, embedding, model_version)
        rows = get_db().execute(
            "SELECT TOP 1 embedding_id FROM FaceEmbeddings "
            "WHERE student_id = ? AND is_active = 1 ORDER BY created_at DESC",
            (student_id,)
        )
        return rows[0][0] if rows else -1

    def load_all_to_cache(self) -> EmbeddingCache:
        """
        SP sp_GetAllEmbeddings trả về 4 cols:
            (student_id, student_code, full_name, embedding_vector)
        """
        rows = get_db().execute(
            """
            SELECT s.student_id, s.student_code, s.full_name, fe.embedding_vector, s.class_id, c.class_name, c.class_code
            FROM FaceEmbeddings fe
            JOIN Students s ON s.student_id = fe.student_id
            LEFT JOIN Classes c ON c.class_id = s.class_id
            WHERE fe.is_active = 1
            ORDER BY fe.student_id
            """
        )

        if not rows:
            logger.warning("Không có embedding nào trong database!")
            return EmbeddingCache()

        cache = EmbeddingCache()
        vecs  = []
        for row in rows:
            student_id, student_code, full_name, emb_bytes, class_id, class_name, class_code = row[:7]
            
            vec = np.frombuffer(emb_bytes, dtype=np.float32).copy()
            if vec.shape[0] != 512:
                logger.warning(f"Skip student={student_id}: shape={vec.shape}")
                continue
            cache.student_ids.append(int(student_id))
            cache.student_codes.append(str(student_code or ""))
            cache.full_names.append(str(full_name or ""))
            cache.class_ids.append(int(class_id or 0))
            cache.class_names.append(str(class_name or ""))
            cache.class_codes.append(str(class_code or ""))
            vecs.append(vec)

        if vecs:
            mat   = np.vstack(vecs).astype(np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            cache.embeddings = mat / np.maximum(norms, 1e-8)

        logger.success(f"Loaded {cache.size} embeddings vào RAM")
        return cache

    def get_all_active(self) -> list:
        try:
            rows = get_db().call_procedure("sp_GetAllEmbeddings")
            return rows if rows else []
        except Exception:
            rows = get_db().execute(
                """
                SELECT s.student_id, s.student_code, s.full_name, fe.embedding_vector
                FROM FaceEmbeddings fe
                JOIN Students s ON s.student_id = fe.student_id
                WHERE fe.is_active = 1
                """
            )
            return rows or []

    def has_embedding(self, student_id: int) -> bool:
        rows = get_db().execute(
            "SELECT COUNT(*) FROM FaceEmbeddings WHERE student_id = ? AND is_active = 1",
            (student_id,)
        )
        return bool(rows and rows[0][0] > 0)

    def delete_by_student(self, student_id: int) -> bool:
        get_db().execute(
            "UPDATE FaceEmbeddings SET is_active = 0 WHERE student_id = ?",
            (student_id,), commit=True,
        )
        return True


# ══════════════════════════════════════════════
#  CAMERA REPOSITORY
# ══════════════════════════════════════════════
class CameraRepository:
    _SQL = """
        SELECT camera_id, camera_name, location_desc,
               rtsp_url, ip_address, resolution, is_active
        FROM Cameras
    """

    def get_all(self, active_only: bool = True) -> list:
        if active_only:
            rows = get_db().execute(f"{self._SQL} WHERE is_active = 1 ORDER BY camera_name")
        else:
            rows = get_db().execute(f"{self._SQL} ORDER BY camera_name")
        return [Camera(*r) for r in rows]

    def get_by_id(self, camera_id: int) -> Optional[Camera]:
        rows = get_db().execute(f"{self._SQL} WHERE camera_id = ?", (camera_id,))
        return Camera(*rows[0]) if rows else None

    def create(self, camera_name: str, location_desc: str = None,
               rtsp_url: str = None, ip_address: str = None,
               resolution: str = "1280x720") -> int:
        rows = get_db().execute(
            """
            INSERT INTO Cameras (camera_name, location_desc, rtsp_url, ip_address, resolution)
            OUTPUT INSERTED.camera_id VALUES (?, ?, ?, ?, ?)
            """,
            (camera_name, location_desc, rtsp_url, ip_address, resolution),
            commit=True,
        )
        return rows[0][0] if rows else -1

    def update(self, camera_id: int, **kwargs) -> bool:
        allowed = {"camera_name", "location_desc", "rtsp_url", "ip_address", "resolution", "is_active"}
        cols = {k: v for k, v in kwargs.items() if k in allowed}
        if not cols: return False
        
        set_clause = ", ".join(f"{k} = ?" for k in cols)
        params = list(cols.values()) + [camera_id]
        get_db().execute(f"UPDATE Cameras SET {set_clause} WHERE camera_id = ?", tuple(params), commit=True)
        return True

    def delete(self, camera_id: int) -> bool:
        get_db().execute("DELETE FROM Cameras WHERE camera_id = ?", (camera_id,), commit=True)
        return True


# ══════════════════════════════════════════════
#  SESSION REPOSITORY
# ══════════════════════════════════════════════
def _row_to_session(r) -> AttendanceSession:
    """13 cols: 11 DB + class_name + class_code."""
    return AttendanceSession(
        session_id    = r[0],
        session_code  = r[1],
        class_id      = r[2],
        subject_name  = r[3],
        session_date  = r[4],
        start_time    = r[5],
        end_time      = r[6],
        status        = r[7]  or "PENDING",
        present_count = int(r[8]  or 0),
        absent_count  = int(r[9]  or 0),
        created_at    = r[10],
        class_name    = r[11],
        class_code    = r[12] or "",
    )


class SessionRepository:

    def get_all(self, limit: int = 200) -> list:
        rows = get_db().execute(
            """
            SELECT TOP (?) s.session_id, s.session_code, s.class_id, s.subject_name,
                   s.session_date, s.start_time, s.end_time,
                   s.status, s.present_count, s.absent_count, s.created_at,
                   c.class_name, c.class_code
            FROM AttendanceSessions s
            LEFT JOIN Classes c ON c.class_id = s.class_id
            ORDER BY s.session_id DESC
            """,
            (limit,)
        )
        return [_row_to_session(r) for r in rows]

    def get_by_id(self, session_id: int) -> Optional[AttendanceSession]:
        rows = get_db().execute(
            """
            SELECT s.session_id, s.session_code, s.class_id, s.subject_name,
                   s.session_date, s.start_time, s.end_time,
                   s.status, s.present_count, s.absent_count, s.created_at,
                   c.class_name, c.class_code
            FROM AttendanceSessions s
            LEFT JOIN Classes c ON c.class_id = s.class_id
            WHERE s.session_id = ?
            """,
            (session_id,)
        )
        return _row_to_session(rows[0]) if rows else None

    def get_recent(self, limit: int = 20) -> list:
        rows = get_db().execute(
            """
            SELECT TOP (?) s.session_id, s.session_code, s.class_id, s.subject_name,
                   s.session_date, s.start_time, s.end_time,
                   s.status, s.present_count, s.absent_count, s.created_at,
                   c.class_name, c.class_code
            FROM AttendanceSessions s
            LEFT JOIN Classes c ON c.class_id = s.class_id
            ORDER BY s.session_id DESC
            """,
            (limit,)
        )
        return [_row_to_session(r) for r in rows]

    def create(self, class_id: int, subject_name: str,
               session_date=None, created_by: str = None) -> int:
        return self.create_session(class_id, subject_name, session_date)

    def create_session(self, class_id: int, subject_name: str,
                       session_date=None) -> int:
        if session_date is None:
            session_date = date.today()
        session_code = (
            f"{class_id}-{session_date.strftime('%Y%m%d')}"
            f"-{datetime.now().strftime('%H%M%S')}"
        )
        rows = get_db().execute(
            """
            INSERT INTO AttendanceSessions
                (session_code, class_id, subject_name, session_date, status)
            OUTPUT INSERTED.session_id
            VALUES (?, ?, ?, ?, 'PENDING')
            """,
            (session_code, class_id, subject_name, session_date),
            commit=True,
        )
        session_id = rows[0][0] if rows else -1
        if session_id > 0:
            self._prefill_absent(session_id, class_id)
        logger.info(f"Created session: {session_code} (id={session_id})")
        return session_id

    def _prefill_absent(self, session_id: int, class_id: int):
        get_db().execute(
            """
            INSERT INTO AttendanceRecords (session_id, student_id, status)
            SELECT ?, student_id, 'ABSENT' FROM Students WHERE class_id = ?
            """,
            (session_id, class_id), commit=True,
        )

    def start_session(self, session_id: int) -> bool:
        get_db().execute(
            "UPDATE AttendanceSessions SET status='ACTIVE', start_time=? WHERE session_id=?",
            (datetime.now(), session_id), commit=True,
        )
        logger.info(f"Session {session_id} → ACTIVE")
        return True

    def end_session(self, session_id: int) -> bool:
        get_db().execute(
            """
            UPDATE AttendanceSessions SET
                status        = 'COMPLETED', end_time = ?,
                present_count = (SELECT COUNT(*) FROM AttendanceRecords
                                 WHERE session_id = ? AND status = 'PRESENT'),
                absent_count  = (SELECT COUNT(*) FROM AttendanceRecords
                                 WHERE session_id = ? AND status = 'ABSENT')
            WHERE session_id = ?
            """,
            (datetime.now(), session_id, session_id, session_id), commit=True,
        )
        logger.info(f"Session {session_id} → COMPLETED")
        return True

    def update_status(self, session_id: int, status: str) -> bool:
        get_db().execute(
            "UPDATE AttendanceSessions SET status = ? WHERE session_id = ?",
            (status, session_id), commit=True,
        )
        return True

    def update_present_count(self, session_id: int) -> int:
        rows = get_db().execute(
            """
            UPDATE AttendanceSessions
            SET present_count = (
                SELECT COUNT(*) FROM AttendanceRecords
                WHERE session_id = ? AND status = 'PRESENT'
            )
            OUTPUT INSERTED.present_count
            WHERE session_id = ?
            """,
            (session_id, session_id), commit=True,
        )
        return rows[0][0] if rows else 0


# ══════════════════════════════════════════════
#  ATTENDANCE RECORD REPOSITORY
# ══════════════════════════════════════════════
class AttendanceRecordRepository:

    def record_attendance(self, session_id: int, student_id: int,
                          recognition_score: float,
                          snapshot_path: str = None,
                          camera_id: int = None) -> bool:
        """SP nhận đúng 3 params: @session_id, @student_id, @score."""
        try:
            get_db().call_procedure(
                "sp_RecordAttendance",
                (session_id, student_id, recognition_score)
            )
            get_db().execute(
                """
                UPDATE AttendanceSessions
                SET present_count = (
                    SELECT COUNT(*) FROM AttendanceRecords
                    WHERE session_id = ? AND status = 'PRESENT'
                ) WHERE session_id = ?
                """,
                (session_id, session_id), commit=True,
            )
            logger.debug(f"Attendance: s={session_id} u={student_id} score={recognition_score:.3f}")
            return True
        except Exception as e:
            logger.error(f"record_attendance error: {e}")
            return False

    def upsert(self, session_id: int, student_id: int,
               status: str, check_in_time=None,
               recognition_score: float = 0) -> int:
        get_db().execute(
            """
            MERGE AttendanceRecords AS target
            USING (SELECT ? AS sid, ? AS uid) AS src
                ON target.session_id = src.sid AND target.student_id = src.uid
            WHEN MATCHED THEN
                UPDATE SET status = ?, check_in_time = ?, recognition_score = ?
            WHEN NOT MATCHED THEN
                INSERT (session_id, student_id, status, check_in_time, recognition_score)
                VALUES (?, ?, ?, ?, ?);
            """,
            (session_id, student_id,
             status, check_in_time, recognition_score,
             session_id, student_id, status, check_in_time, recognition_score),
            commit=True,
        )
        rows = get_db().execute(
            "SELECT record_id FROM AttendanceRecords WHERE session_id=? AND student_id=?",
            (session_id, student_id)
        )
        return rows[0][0] if rows else -1

    def get_session_report(self, session_id: int) -> list:
        rows = get_db().execute(
            """
            SELECT s.student_code, s.full_name, c.class_name,
                   ar.status, ar.check_in_time, ar.recognition_score,
                   ar.snapshot_path, s.gender, ar.camera_id, s.student_id
            FROM AttendanceRecords ar
            INNER JOIN Students s ON s.student_id = ar.student_id
            LEFT  JOIN Classes  c ON c.class_id   = s.class_id
            WHERE ar.session_id = ?
            ORDER BY ar.status DESC, ar.check_in_time
            """,
            (session_id,)
        )
        result = []
        for r in rows:
            t = r[4]
            time_str = t.strftime("%H:%M:%S") if hasattr(t, "strftime") else str(t or "")
            result.append({
                "student_code":      r[0] or "",
                "full_name":         r[1] or "",
                "class_name":        r[2] or "",
                "status":            r[3] or "ABSENT",
                "check_in_time":     time_str,
                "recognition_score": float(r[5] or 0),
                "recognition_method": "FACE",
                "gender":            r[7] or "",
                "camera_id":         r[8] or 1,
                "student_id":        r[9] or 0,
            })
        return result

    def is_already_recorded(self, session_id: int, student_id: int) -> bool:
        rows = get_db().execute(
            "SELECT status FROM AttendanceRecords WHERE session_id=? AND student_id=?",
            (session_id, student_id)
        )
        return bool(rows and rows[0][0] == "PRESENT")

    def get_present_list(self, session_id: int) -> list:
        rows = get_db().execute(
            """
            SELECT s.student_code, s.full_name, ar.check_in_time, ar.recognition_score
            FROM AttendanceRecords ar
            INNER JOIN Students s ON s.student_id = ar.student_id
            WHERE ar.session_id = ? AND ar.status = 'PRESENT'
            ORDER BY ar.check_in_time
            """,
            (session_id,)
        )
        return [{"code": r[0], "name": r[1], "time": r[2], "score": r[3]} for r in rows]


# ─────────────────────────────────────────────
#  Singleton instances
# ─────────────────────────────────────────────
class_repo     = ClassRepository()
student_repo   = StudentRepository()
embedding_repo = FaceEmbeddingRepository()
camera_repo    = CameraRepository()
session_repo   = SessionRepository()
record_repo    = AttendanceRecordRepository()