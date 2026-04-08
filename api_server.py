"""
api_server.py
Máy chủ Trung tâm - Đón nhận dữ liệu khuôn mặt (Vector) từ các Edge Box
"""
import uvicorn
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from contextlib import asynccontextmanager
import numpy as np
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import các module từ project của bạn
from services.embedding_cache_manager import cache_manager
from database.repositories import student_repo, record_repo
from database.connection import get_db
from config import ai_config

# ─────────────────────────────────────────────
#  1. KHỞI ĐỘNG SERVER & LOAD DATABASE (CHUẨN MỚI)
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 API Server đang khởi động...")
    # SỬA LỖI Ở ĐÂY: Dùng hàm load() theo đúng file embedding_cache_manager.py
    cache_manager.load()
    cache = cache_manager.get_cache()
    if not cache.is_empty:
        logger.info(f"✅ Đã tải thành công {len(cache.embeddings)} khuôn mặt vào RAM.")
    else:
        logger.warning("⚠️ Database hiện chưa có học viên nào!")
    
    yield # Server bắt đầu chạy tại đây
    
    logger.info("🛑 API Server đang tắt...")

app = FastAPI(title="FaceAttend API Server", lifespan=lifespan)

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
#  3. ĐỊNH NGHĨA CẤU TRÚC GÓI TIN TỪ EDGE BOX
# ─────────────────────────────────────────────
class AttendancePayload(BaseModel):
    camera_id: str
    timestamp: str
    embedding: list[float]  # Dải 512 con số
    liveness_score: float = 1.0 # Điểm chống gian lận

# ─────────────────────────────────────────────
#  4. API XỬ LÝ ĐIỂM DANH CHÍNH
# ─────────────────────────────────────────────
@app.post("/api/attendance")
async def receive_attendance(payload: AttendancePayload, api_key: str = Security(verify_api_key)):
    try:
        incoming_vector = np.array(payload.embedding, dtype=np.float32)
        if len(incoming_vector) != 512:
            raise HTTPException(status_code=400, detail="Vector không hợp lệ")

        cache = cache_manager.get_cache()
        if cache.is_empty:
            return {"status": "ignored", "message": "CSDL trống"}

        # SO KHỚP VECTOR BẰNG COSINE SIMILARITY
        similarities = np.dot(cache.embeddings, incoming_vector) / (
            np.linalg.norm(cache.embeddings, axis=1) * np.linalg.norm(incoming_vector)
        )
        
        best_idx = np.argmax(similarities)
        best_score = float(similarities[best_idx])

        # KIỂM TRA NGƯỠNG NHẬN DIỆN
        if best_score >= ai_config.recognition_threshold:
            student_id = cache.student_ids[best_idx]
            student = student_repo.get_by_id(student_id)
            
            if not student:
                return {"status": "error", "message": "Học viên không tồn tại trong DB"}

            # Truy vấn xem lớp của sinh viên này có session nào đang 'ACTIVE' không
            active_session = get_db().execute(
                "SELECT TOP 1 session_id FROM AttendanceSessions WHERE class_id = ? AND status = 'ACTIVE' ORDER BY created_at DESC",
                (student.class_id,)
            )

            if not active_session:
                logger.warning(f"⚠️ Nhận diện được {student.full_name} nhưng lớp chưa có phiên điểm danh nào đang mở!")
                return {
                    "status": "warning", 
                    "message": "Không có phiên học nào đang mở cho lớp này."
                }

            session_id = active_session[0][0]

            # Ghi nhận vào DB
            success = record_repo.record_attendance(
                session_id=session_id,
                student_id=student_id,
                recognition_score=best_score,
                camera_id=1 
            )
            
            if success:
                logger.success(f"✅ Đã điểm danh: [{student.student_code}] {student.full_name} (Score: {best_score:.2f})")
                return {
                    "status": "success", 
                    "student_code": student.student_code,
                    "full_name": student.full_name,
                    "similarity": best_score
                }
            else:
                return {"status": "error", "message": "Lỗi khi ghi vào Database"}
        else:
            logger.warning(f"❌ Người lạ (Score cao nhất chỉ đạt {best_score:.2f})")
            return {"status": "unknown", "message": "Không nhận diện được khuôn mặt"}

    except Exception as e:
        logger.error(f"Lỗi API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)