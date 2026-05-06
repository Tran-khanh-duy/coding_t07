import base64
from datetime import datetime
from typing import Optional, Any
from fastapi import APIRouter, Security, HTTPException, Query, Response
from pydantic import BaseModel
from loguru import logger

from core.security import verify_device_access
from core.state_manager import state_manager
from database.repositories import camera_repo

router = APIRouter(prefix="/api/system", tags=["System & Cameras"])

class CommandPayload(BaseModel):
    command: str
    session_id: Optional[int] = None
    class_id: Optional[int] = None
    target_camera: Optional[str] = None

class FramePayload(BaseModel):
    image_b64: str
    detections: Optional[list] = []
    camera_id: str = "CAM_01"

class EdgeStatusPayload(BaseModel):
    device_name: str
    camera_status: dict[str, Any]
    ip_address: Optional[str] = "Unknown"
    timestamp: str

@router.get("/command")
async def get_system_command(api_key: str = Security(verify_device_access)):
    """Mini PC polling lấy lệnh từ Server."""
    cameras = camera_repo.get_all(active_only=False)
    all_rtsp = [c.rtsp_url for c in cameras if c.rtsp_url]
    
    for dev_status in state_manager.get_all_edge_status().values():
        cam_status = dev_status.get("camera_status", {})
        for cam_id, info in cam_status.items():
            if isinstance(info, dict):
                src = info.get("source")
                if src and src not in all_rtsp:
                    all_rtsp.append(src)
    
    cmd_state = state_manager.get_command_state()
    return {
        "command": cmd_state["command"],
        "session_id": cmd_state["session_id"],
        "class_id": cmd_state["class_id"],
        "target_camera": cmd_state["target_camera"],
        "all_cameras": all_rtsp
    }

@router.post("/command")
async def set_system_command(
    payload: CommandPayload,
    api_key: str = Security(verify_device_access)
):
    """Server UI phát lệnh cho Mini PC."""
    state_manager.set_command(
        command=payload.command,
        session_id=payload.session_id,
        class_id=payload.class_id,
        target_camera=payload.target_camera
    )
    cmd_state = state_manager.get_command_state()
    logger.info(f"📡 Lệnh Hệ Thống thay đổi -> COMMAND: {cmd_state['command']}")
    return {"status": "ok", "state": cmd_state['command']}

@router.post("/frame")
async def upload_frame(
    payload: FramePayload,
    api_key: str = Security(verify_device_access)
):
    """Mini PC upload khung hình JPEG (base64) lên server."""
    try:
        img_data = base64.b64decode(payload.image_b64)
        if len(img_data) < 100:
             logger.warning(f"⚠️ Nhận ảnh quá nhỏ ({len(img_data)} bytes) từ {payload.camera_id}")
             
        state_manager.set_latest_frame(payload.camera_id, img_data, payload.detections)
        return {"status": "ok", "received_bytes": len(img_data)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/frame")
async def get_frame(camera_id: str = Query(..., description="ID của camera cần lấy hình")):
    """UI lấy khung hình mới nhất từ Mini PC."""
    import json
    frame, detections = state_manager.get_latest_frame(camera_id)
    if frame is None:
        available = state_manager.get_available_cameras()
        raise HTTPException(status_code=404, detail=f"No frame for {camera_id}. Available: {available}")
        
    det_json = json.dumps(detections)
    det_b64 = base64.b64encode(det_json.encode()).decode()

    return Response(
        content=frame, 
        media_type="image/jpeg",
        headers={"X-Face-Detections": det_b64}
    )

@router.post("/edge_status")
async def update_edge_status(
    payload: EdgeStatusPayload,
    api_key: str = Security(verify_device_access)
):
    """Mini PC báo cáo danh sách camera hiện có."""
    state_manager.update_edge_status(payload.device_name, {
        "camera_status": payload.camera_status,
        "ip_address": payload.ip_address,
        "last_seen": datetime.now().timestamp()
    })
    logger.info(f"📡 Mini PC '{payload.device_name}' đã báo danh trạng thái.")
    return {"status": "ok"}

@router.get("/edge_status")
async def get_edge_status():
    """UI lấy danh sách camera động từ các Mini PC."""
    return state_manager.get_all_edge_status()
