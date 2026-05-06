from fastapi import APIRouter, Security, HTTPException
from loguru import logger
from datetime import datetime
import time
import psutil

from core.security import verify_device_access
from core.state_manager import state_manager
from database.connection import db
from database.repositories import student_repo, camera_repo
from services.embedding_service import embedding_service

router = APIRouter(tags=["Admin & Dashboard"])

@router.get("/api/health")
async def health_check():
    """Kiểm tra tình trạng server."""
    return {"status": "ok", "time": datetime.now().isoformat()}

@router.get("/admin/system-status")
async def system_status(api_key: str = Security(verify_device_access)):
    """Monitoring endpoint cho dashboard nội bộ."""
    try:
        # Lấy CPU & RAM
        cpu_usage = psutil.cpu_percent(interval=0.1)
        ram_usage = psutil.virtual_memory().percent
        
        # Redis metrics & queue size
        redis_start = time.time()
        redis_ping = state_manager.redis.ping()
        redis_latency = (time.time() - redis_start) * 1000
        queue_size = state_manager.redis.llen("queue:attendance")
        
        # DB Latency
        db_start = time.time()
        db.execute("SELECT 1")
        db_latency = (time.time() - db_start) * 1000
        
        # Edge/Camera status
        online_cams = 0
        total_cams = 0
        edge_status = state_manager.get_all_edge_status()
        for dev_status in edge_status.values():
            cam_status = dev_status.get("camera_status", {})
            for cam_id, info in cam_status.items():
                total_cams += 1
                if isinstance(info, dict) and info.get("is_active"):
                    online_cams += 1
                    
        # Nếu chưa có edge heartbeat nào, lấy từ DB
        if total_cams == 0:
            db_cams = camera_repo.get_all()
            total_cams = len(db_cams)
            
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": round(cpu_usage, 2),
            "ram_percent": round(ram_usage, 2),
            "queue_size": queue_size,
            "cameras": {
                "online": online_cams,
                "total": total_cams
            },
            "latency_ms": {
                "database": round(db_latency, 2),
                "redis": round(redis_latency, 2)
            },
            "edge_devices_online": len(edge_status)
        }
    except Exception as e:
        logger.error(f"Lỗi API /admin/system-status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/dashboard/stats")
async def get_dashboard_stats(api_key: str = Security(verify_device_access)):
    """Trả về thống kê cho Dashboard: Số lượng học viên, Camera Online/Offline."""
    try:
        students = student_repo.get_all()
        student_count = len(students)
        
        online_cams = 0
        total_cams = 0
        for dev_status in state_manager.get_all_edge_status().values():
            cam_status = dev_status.get("camera_status", {})
            for cam_id, info in cam_status.items():
                total_cams += 1
                if isinstance(info, dict) and info.get("is_active"):
                    online_cams += 1
                    
        offline_cams = total_cams - online_cams
        
        if total_cams == 0:
            db_cams = camera_repo.get_all()
            total_cams = len(db_cams)
            offline_cams = total_cams
            
        return {
            "status": "success",
            "student_count": student_count,
            "total_cameras": total_cams,
            "online_cameras": online_cams,
            "offline_cameras": offline_cams
        }
    except Exception as e:
        logger.error(f"Lỗi API /dashboard/stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/reload-cache")
async def reload_cache(api_key: str = Security(verify_device_access)):
    """Force reload embedding cache từ DB."""
    try:
        embedding_service.reload()
        new_version = embedding_service.version
        size = embedding_service.size
        logger.info(f"📢 Embedding version → {new_version} (có học viên mới)")
        return {
            "status": "ok",
            "message": f"Đã reload {size} embeddings",
            "count": size,
            "embedding_version": new_version,
        }
    except Exception as e:
        logger.error(f"Lỗi reload cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))
