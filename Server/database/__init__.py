#database/__init__.py

from .connection import db, get_db, DatabaseConnection
from .models import (
    Class, Student, FaceEmbedding,
    AttendanceSession, AttendanceRecord,
    Camera, EmbeddingCache,
)
from .repositories import (
    class_repo, student_repo, embedding_repo,
    camera_repo, session_repo, record_repo,
    ClassRepository, StudentRepository,
    FaceEmbeddingRepository, CameraRepository,
    SessionRepository, AttendanceRecordRepository,
)

__all__ = [
    "db", "get_db", "DatabaseConnection",
    "Class", "Student", "FaceEmbedding",
    "AttendanceSession", "AttendanceRecord",
    "Camera", "EmbeddingCache",
    "class_repo", "student_repo", "embedding_repo",
    "camera_repo", "session_repo", "record_repo",
    "ClassRepository", "StudentRepository",
    "FaceEmbeddingRepository", "CameraRepository",
    "SessionRepository", "AttendanceRecordRepository",
]