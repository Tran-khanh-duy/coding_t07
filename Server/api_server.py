"""
api_server.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Máy chủ Trung tâm (Server) — FastAPI
Cung cấp API cho các Edge Box (Mini PC) kết nối:
  - GET  /api/health        → Kiểm tra trạng thái server
  - GET  /api/students      → Danh sách học viên
  - GET  /api/embeddings    → Tải embedding vectors để nhận diện
  - GET  /api/sessions/active → Danh sách phiên điểm danh đang mở
  - POST /api/attendance    → Nhận kết quả điểm danh từ Edge

Khởi động:
    python api_server.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import uvicorn
import base64
import threading
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Security, Query, Response
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from loguru import logger

# Biến toàn cục quản lý trạng thái phát lệnh cho Mini PC
class SystemState:
    command = "STOP"
    session_id = None
    class_id = None
    target_camera = None
    
    # Mới: Lưu trữ khung hình từ Mini PC
    latest_frame: Optional[bytes] = None
    frame_timestamp: float = 0.0
    frame_lock = threading.Lock()

system_state = SystemState()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import các module từ project
from services.embedding_cache_manager import cache_manager
from database.repositories import student_repo, record_repo, session_repo, embedding_repo, class_repo
from database.connection import get_db
from config import ai_config

# ─────────────────────────────────────────────
#  1. KHỞI ĐỘNG SERVER & LOAD DATABASE
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 API Server đang khởi động...")
    cache_manager.load()
    cache = cache_manager.get_cache()
    if not cache.is_empty:
        logger.info(f"✅ Đã tải thành công {len(cache.embeddings)} khuôn mặt vào RAM.")
    else:
        logger.warning("⚠️ Database hiện chưa có học viên nào!")
    
    yield
    logger.info("🛑 API Server đang tắt...")

app = FastAPI(
    title="FaceAttend API Server",
    description="API trung tâm cho hệ thống điểm danh khuôn mặt",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — cho phép Edge Box từ bất kỳ IP nào trong LAN
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  2. CẤU HÌNH BẢO MẬT (API KEY)
# ─────────────────────────────────────────────
SECRET_API_KEY = "faceattend_secret_2026"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != SECRET_API_KEY:
        raise HTTPException(status_code=403, detail="Từ chối truy cập! Sai API Key.")
    return api_key

# ─────────────────────────────────────────────
#  3. SCHEMAS (Cấu trúc dữ liệu)
# ─────────────────────────────────────────────
class AttendancePayload(BaseModel):
    camera_id: str
    timestamp: str
    embedding: list[float]  # Vector 512D
    liveness_score: float = 1.0
    liveness_checked: bool = False

class AttendanceResponse(BaseModel):
    status: str
    message: str = ""
    student_code: Optional[str] = None
    full_name: Optional[str] = None
    class_name: Optional[str] = None
    similarity: Optional[float] = None
    session_id: Optional[int] = None

# ─────────────────────────────────────────────
#  4. API ENDPOINTS
# ─────────────────────────────────────────────

# ── Health Check ──────────────────────────────
@app.get("/api/health")
async def health_check():
    """Kiểm tra server còn sống không — Edge Box gọi định kỳ."""
    cache = cache_manager.get_cache()
    try:
        db_ok = get_db().test_connection()
    except Exception:
        db_ok = False

    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "database": "connected" if db_ok else "disconnected",
        "total_embeddings": cache.size,
        "version": "2.0.0",
    }

# ── Lệnh Hệ thống (Master-Slave) ──────────────
class CommandPayload(BaseModel):
    command: str
    session_id: Optional[int] = None
    class_id: Optional[int] = None
    target_camera: Optional[str] = None

@app.get("/api/system/command")
async def get_system_command(api_key: str = Security(verify_api_key)):
    """Mini PC polling lấy lệnh từ Server."""
    return {
        "command": system_state.command,
        "session_id": system_state.session_id,
        "class_id": system_state.class_id,
        "target_camera": system_state.target_camera
    }

@app.post("/api/system/command")
async def set_system_command(
    payload: CommandPayload,
    api_key: str = Security(verify_api_key)
):
    """Server UI phát lệnh cho Mini PC."""
    system_state.command = payload.command
    system_state.session_id = payload.session_id
    system_state.class_id = payload.class_id
    system_state.target_camera = payload.target_camera
    
    logger.info(f"📡 Lệnh Hệ Thống thay đổi -> COMMAND: {system_state.command} | SESSION: {system_state.session_id} | CAM: {system_state.target_camera}")
    return {"status": "ok", "state": system_state.command}

# ── Truyền tải khung hình (Remote Camera) ──────
class FramePayload(BaseModel):
    image_b64: str

@app.post("/api/system/frame")
async def upload_frame(
    payload: FramePayload,
    api_key: str = Security(verify_api_key)
):
    """Mini PC upload khung hình JPEG (base64) lên server."""
    try:
        img_data = base64.b64decode(payload.image_b64)
        with system_state.frame_lock:
            system_state.latest_frame = img_data
            system_state.frame_timestamp = datetime.now().timestamp()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/system/frame")
async def get_frame():
    """UI lấy khung hình mới nhất từ Mini PC."""
    with system_state.frame_lock:
        if system_state.latest_frame is None:
            raise HTTPException(status_code=404, detail="No frame available")
        
        # Có thể thêm kiểm tra stale nếu cần (ví dụ > 5s thì coi như mất kết nối)
        return Response(content=system_state.latest_frame, media_type="image/jpeg")

# ── Lấy danh sách học viên ────────────────────
@app.get("/api/students")
async def get_students(
    class_id: Optional[int] = Query(None, description="Lọc theo lớp"),
    api_key: str = Security(verify_api_key),
):
    """Trả về danh sách học viên (metadata, không có embedding)."""
    try:
        students = student_repo.get_all(class_id=class_id)
        return {
            "status": "ok",
            "count": len(students),
            "students": [
                {
                    "student_id": s.student_id,
                    "student_code": s.student_code,
                    "full_name": s.full_name,
                    "class_id": s.class_id,
                    "class_name": s.class_name,
                    "face_enrolled": s.face_enrolled,
                }
                for s in students
            ],
        }
    except Exception as e:
        logger.error(f"Lỗi API /students: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Lấy tất cả embedding vectors ──────────────
@app.get("/api/embeddings")
async def get_embeddings(api_key: str = Security(verify_api_key)):
    """
    Trả về TOÀN BỘ embedding vectors để Edge load vào RAM.
    Dữ liệu được mã hoá base64 để gửi qua JSON hiệu quả.
    
    Response format:
    {
        "count": 50,
        "students": [
            {
                "student_id": 1,
                "student_code": "SV001",
                "full_name": "Nguyen Van A",
                "class_id": 1,
                "class_name": "CNTT01",
                "class_code": "CNTT01",
                "embedding_b64": "base64_encoded_512_float32..."
            }, ...
        ]
    }
    """
    try:
        cache = cache_manager.get_cache()
        if cache.is_empty:
            return {"status": "ok", "count": 0, "students": []}

        students = []
        for i in range(cache.size):
            # Encode embedding thành base64 string
            emb_bytes = cache.embeddings[i].astype(np.float32).tobytes()
            emb_b64 = base64.b64encode(emb_bytes).decode("ascii")

            students.append({
                "student_id": cache.student_ids[i],
                "student_code": cache.student_codes[i],
                "full_name": cache.full_names[i],
                "class_id": cache.class_ids[i],
                "class_name": cache.class_names[i],
                "class_code": cache.class_codes[i],
                "embedding_b64": emb_b64,
            })

        logger.info(f"📤 Edge yêu cầu embeddings: trả về {len(students)} học viên")
        return {"status": "ok", "count": len(students), "students": students}

    except Exception as e:
        logger.error(f"Lỗi API /embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Lấy danh sách phiên điểm danh đang ACTIVE ──
@app.get("/api/sessions/active")
async def get_active_sessions(api_key: str = Security(verify_api_key)):
    """Trả về các phiên điểm danh đang mở (status = ACTIVE)."""
    try:
        all_sessions = session_repo.get_all(limit=50)
        active = [s for s in all_sessions if s.status == "ACTIVE"]

        return {
            "status": "ok",
            "count": len(active),
            "sessions": [
                {
                    "session_id": s.session_id,
                    "session_code": s.session_code,
                    "class_id": s.class_id,
                    "class_name": s.class_name or "",
                    "class_code": s.class_code or "",
                    "subject_name": s.subject_name,
                    "session_date": str(s.session_date) if s.session_date else "",
                    "start_time": s.start_time.isoformat() if s.start_time else "",
                    "present_count": s.present_count,
                }
                for s in active
            ],
        }
    except Exception as e:
        logger.error(f"Lỗi API /sessions/active: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Nhận kết quả điểm danh từ Edge ────────────
@app.post("/api/attendance", response_model=AttendanceResponse)
async def receive_attendance(
    payload: AttendancePayload,
    api_key: str = Security(verify_api_key),
):
    """
    Edge Box gửi embedding vector + liveness score.
    Server so khớp và ghi nhận điểm danh.
    """
    try:
        incoming_vector = np.array(payload.embedding, dtype=np.float32)
        if len(incoming_vector) != 512:
            raise HTTPException(status_code=400, detail="Vector không hợp lệ (cần 512 chiều)")

        # Kiểm tra Anti-Spoofing
        if payload.liveness_checked and payload.liveness_score < 0.80:
            logger.warning(f"❌ Edge báo SPOOF! Score: {payload.liveness_score:.3f}")
            return AttendanceResponse(
                status="rejected",
                message=f"Phát hiện giả mạo (Liveness: {payload.liveness_score:.2f})",
            )

        cache = cache_manager.get_cache()
        if cache.is_empty:
            return AttendanceResponse(status="ignored", message="CSDL trống — chưa có học viên")

        # SO KHỚP VECTOR BẰNG COSINE SIMILARITY
        norm_in = np.linalg.norm(incoming_vector)
        if norm_in < 1e-8:
            return AttendanceResponse(status="error", message="Vector embedding rỗng")
        incoming_vector = incoming_vector / norm_in

        similarities = cache.embeddings @ incoming_vector  # Cache đã normalized sẵn

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        # KIỂM TRA NGƯỠNG NHẬN DIỆN
        if best_score >= ai_config.recognition_threshold:
            student_id = cache.student_ids[best_idx]
            student_code = cache.student_codes[best_idx]
            full_name = cache.full_names[best_idx]
            class_name = cache.class_names[best_idx]
            class_id = cache.class_ids[best_idx]

            # Tìm session ACTIVE cho lớp này
            all_sessions = session_repo.get_all(limit=50)
            active_session = None
            for s in all_sessions:
                if s.class_id == class_id and s.status == "ACTIVE":
                    active_session = s
                    break

            if not active_session:
                logger.warning(
                    f"⚠️ Nhận diện [{student_code}] {full_name} "
                    f"nhưng lớp chưa có phiên điểm danh nào đang mở!"
                )
                return AttendanceResponse(
                    status="no_session",
                    message="Không có phiên điểm danh đang mở cho lớp này",
                    student_code=student_code,
                    full_name=full_name,
                    class_name=class_name,
                    similarity=best_score,
                )

            session_id = active_session.session_id

            # Kiểm tra đã điểm danh chưa
            already = record_repo.is_already_recorded(session_id, student_id)
            if already:
                return AttendanceResponse(
                    status="duplicate",
                    message="Đã điểm danh rồi",
                    student_code=student_code,
                    full_name=full_name,
                    class_name=class_name,
                    similarity=best_score,
                    session_id=session_id,
                )

            # Ghi nhận vào DB
            success = record_repo.record_attendance(
                session_id=session_id,
                student_id=student_id,
                recognition_score=best_score,
                camera_id=1,
            )

            if success:
                logger.success(
                    f"✅ Đã điểm danh: [{student_code}] {full_name} "
                    f"(Score: {best_score:.2f}) | Session: {session_id}"
                )
                return AttendanceResponse(
                    status="success",
                    message="Điểm danh thành công",
                    student_code=student_code,
                    full_name=full_name,
                    class_name=class_name,
                    similarity=best_score,
                    session_id=session_id,
                )
            else:
                return AttendanceResponse(status="error", message="Lỗi khi ghi vào Database")
        else:
            logger.warning(f"❌ Người lạ (Score cao nhất: {best_score:.2f})")
            return AttendanceResponse(
                status="unknown",
                message=f"Không nhận diện được (Score: {best_score:.2f})",
                similarity=best_score,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi API /attendance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Reload cache (Sau khi đăng ký HV mới) ────
@app.post("/api/reload-cache")
async def reload_cache(api_key: str = Security(verify_api_key)):
    """Force reload embedding cache từ DB."""
    try:
        cache_manager.load()
        cache = cache_manager.get_cache()
        return {
            "status": "ok",
            "message": f"Đã reload {cache.size} embeddings",
            "count": cache.size,
        }
    except Exception as e:
        logger.error(f"Lỗi reload cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Log Filtering (Silence /api/system/frame spam) ───
import logging

class EndpointFilter(logging.Filter):
    """Lọc các dòng log chứa endpoint streaming để console sạch hơn."""
    def filter(self, record: logging.LogRecord) -> bool:
        # Nếu log chứa endpoint frame, trả về False để không in ra
        return "/api/system/frame" not in record.getMessage()

# Áp dụng filter cho uvicorn
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())
logging.getLogger("uvicorn").addFilter(EndpointFilter())

# ─────────────────────────────────────────────
#  5. KHỞI ĐỘNG
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  FACEATTEND API SERVER v2.0")
    logger.info("  Lắng nghe tại: http://0.0.0.0:8000")
    logger.info("=" * 60)
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)