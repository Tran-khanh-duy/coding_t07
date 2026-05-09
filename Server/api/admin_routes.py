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
from pydantic import BaseModel
import socket
from concurrent.futures import ThreadPoolExecutor

def _check_port(ip_port: str, timeout=0.8) -> bool:
    if not ip_port or ip_port == "OFFLINE" or ip_port.startswith("127."):
        return False
    try:
        ip = ip_port.split(":")[0]
        port = int(ip_port.split(":")[1]) if ":" in ip_port else 554
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((ip, port))
        return True
    except Exception:
        return False

def get_real_camera_status():
    """Lấy số lượng camera thực tế bằng cách check port 554 song song"""
    total_cams = 0
    ips_to_check = []
    
    edge_status = state_manager.get_all_edge_status()
    for dev_status in edge_status.values():
        cam_status = dev_status.get("camera_status", {})
        for cam_id, info in cam_status.items():
            total_cams += 1
            source = cam_id
            if isinstance(info, dict):
                source = info.get("source", cam_id)
                
            if isinstance(source, str) and source.startswith("rtsp://"):
                try:
                    cam_ip = source.split("@")[1].split("/")[0] if "@" in source else source.split("://")[1].split("/")[0]
                    ips_to_check.append(cam_ip)
                except:
                    pass

    if total_cams == 0:
        db_cams = camera_repo.get_all()
        total_cams = len(db_cams)
        for cam in db_cams:
            if cam.rtsp_url:
                try:
                    cam_ip = cam.rtsp_url.split("@")[1].split("/")[0] if "@" in cam.rtsp_url else cam.rtsp_url.split("://")[1].split("/")[0]
                    ips_to_check.append(cam_ip)
                except:
                    pass

    online_cams = 0
    if ips_to_check:
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(_check_port, ips_to_check))
            online_cams = sum(1 for r in results if r)
            
    return total_cams, online_cams

# Global state lưu trữ tạm thời trên RAM Server
edge_hardware_state = {
    "cpu_percent": 0.0,
    "ram_percent": 0.0,
    "temperature": 0.0,
    "last_updated": 0
}

class HardwareMetrics(BaseModel):
    edge_id: str
    cpu_percent: float
    ram_percent: float
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    temperature: float

router = APIRouter(tags=["Admin & Dashboard"])

@router.get("/api/health")
async def health_check():
    """Kiểm tra tình trạng server."""
    return {"status": "ok", "time": datetime.now().isoformat()}

@router.post("/api/hardware/metrics")
async def update_hardware_metrics(metrics: HardwareMetrics):
    print(f"[API DEBUG] Nhận gói dữ liệu POST từ MiniPC: {metrics.model_dump()}")
    global edge_hardware_state
    edge_hardware_state.update({
        "cpu_percent": metrics.cpu_percent,
        "ram_percent": metrics.ram_percent,
        "ram_used_gb": metrics.ram_used_gb,
        "ram_total_gb": metrics.ram_total_gb,
        "temperature": metrics.temperature,
        "last_updated": time.time()
    })
    return {"status": "success"}

@router.get("/api/hardware/metrics")
async def get_hardware_metrics():
    global edge_hardware_state
    return edge_hardware_state

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
        
        # Edge/Camera status (Dùng socket check thực tế)
        total_cams, online_cams = get_real_camera_status()
        edge_status = state_manager.get_all_edge_status()
            
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
    """Trả về thống kê đầy đủ cho Dashboard: sĩ số, điểm danh, camera, log mới nhất."""
    import time as _time
    t_api_start = _time.time()
    try:
        # 1. Số học viên tổng
        students = student_repo.get_all()
        student_count = len(students)

        # 2. Camera online / offline từ Socket Check
        total_cams, online_cams = get_real_camera_status()

        offline_cams = total_cams - online_cams

        # 3. Điểm danh hôm nay: lấy session ACTIVE hoặc COMPLETED mới nhất trong ngày
        today_str = datetime.now().strftime("%Y-%m-%d")
        rows_session = db.execute(
            """
            SELECT session_id, present_count, absent_count
            FROM AttendanceSessions
            WHERE DATE(session_date) = ? AND status IN ('ACTIVE','COMPLETED')
            ORDER BY session_id DESC LIMIT 1
            """,
            (today_str,)
        )

        present_count = 0
        absent_count  = 0
        latest_logs   = []

        if rows_session:
            sid, present_count, absent_count = rows_session[0]
            present_count = int(present_count or 0)
            absent_count  = int(absent_count  or 0)

            # 4. Log 20 bản ghi PRESENT mới nhất trong session hôm nay
            rows_log = db.execute(
                """
                SELECT hv.MaHV, hv.HoTen, ar.check_in_time,
                       COALESCE(c.camera_name, CONCAT('CAM_', ar.camera_id)) AS cam_label,
                       l.IDLop
                FROM AttendanceRecords ar
                INNER JOIN hocvien hv ON hv.id = ar.student_id
                LEFT  JOIN lop     l  ON l.IDLop = hv.IDLop
                LEFT  JOIN Cameras c  ON c.camera_id = ar.camera_id
                WHERE ar.session_id = ? AND ar.status = 'PRESENT'
                ORDER BY ar.check_in_time DESC
                LIMIT 20
                """,
                (sid,)
            )
            for r in rows_log:
                t = r[2]
                time_str = t.strftime("%H:%M:%S") if hasattr(t, "strftime") else str(t or "")
                latest_logs.append({
                    "id":         str(r[0] or ""),
                    "name":       str(r[1] or ""),
                    "time":       time_str,
                    "camera":     str(r[3] or "CAM"),
                    "class":      str(r[4] or ""),
                    "is_present": True,
                })

        # 5. API Latency (self-measured)
        api_latency_ms = round((_time.time() - t_api_start) * 1000, 2)

        return {
            "status":         "success",
            "student_count":  student_count,
            "present_count":  present_count,
            "absent_count":   absent_count,
            "total_cameras":  total_cams,
            "online_cameras": online_cams,
            "offline_cameras": offline_cams,
            "spoof_warnings": 0,          # Sẽ mở rộng từ state_manager sau
            "latest_logs":    latest_logs,
            "api_latency_ms": api_latency_ms,
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
