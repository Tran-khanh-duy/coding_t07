#__init__.py
from .face_engine import face_engine, FaceEngine, DetectedFace, RecognitionResult
from .embedding_cache_manager import cache_manager, EmbeddingCacheManager
from .camera_manager import camera_manager, CameraManager, CameraStatus, CameraInfo
from .frame_processor import FrameProcessor
from .attendance_service import attendance_service, AttendanceService, AttendanceEvent
from .enrollment_service import enrollment_service, EnrollmentService, EnrollmentResult

__all__ = [
    "face_engine", "FaceEngine", "DetectedFace", "RecognitionResult",
    "cache_manager", "EmbeddingCacheManager",
    "camera_manager", "CameraManager", "CameraStatus", "CameraInfo",
    "FrameProcessor",
    "attendance_service", "AttendanceService", "AttendanceEvent",
    "enrollment_service", "EnrollmentService", "EnrollmentResult",
]
