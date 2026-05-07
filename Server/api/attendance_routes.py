import base64
import json
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Security, HTTPException, Query
from pydantic import BaseModel
from core.logger import logger, attendance_logger

from core.security import verify_device_access
from core.state_manager import state_manager
from core.config import ai_config
from database.repositories import student_repo, session_repo, camera_repo
from services.embedding_service import embedding_service

router = APIRouter(tags=["Attendance & Embeddings"])

class AttendancePayload(BaseModel):
    camera_id: str
    timestamp: str
    embedding: list[float]
    liveness_score: Optional[float] = 1.0
    liveness_checked: Optional[bool] = False

class AttendanceResponse(BaseModel):
    status: str
    message: str
    student_code: Optional[str] = None
    full_name: Optional[str] = None
    class_name: Optional[str] = None
    similarity: Optional[float] = None
    session_id: Optional[int] = None

def _get_valid_student_ids(building: str, floor: str) -> Optional[list[int]]:
    """
    Lay danh sach student_id theo tang/toa nha.
    Schema: hocvien.building (String), hocvien.floor (String).
    Tra None neu khong filter duoc → nhan dien toan bo DB (an toan).
    """
    if not building and not floor:
        return None
    try:
        from database.connection import get_db
        conditions, params = [], []
        if building:
            conditions.append("hv.building = ?")
            params.append(building)
        if floor:
            # floor trong hocvien la String, camera.floor la int
            conditions.append("(hv.floor = ? OR hv.floor = ?)")
            floor_str = str(floor)
            params.extend([floor_str, floor_str.lstrip("0") or "0"])

        sql = f"SELECT hv.id FROM hocvien hv WHERE {' AND '.join(conditions)}"
        rows = get_db().execute(sql, tuple(params))
        if rows:
            ids = [int(r[0]) for r in rows]
            logger.info(f"Floor filter [{building} T{floor}]: {len(ids)} hoc vien")
            return ids

        # Khong tim thay HV nao voi filter nay → tra None de khong bi chan toan bo
        logger.warning(
            f"Khong tim thay HV: building='{building}' floor='{floor}' "
            f"- bo qua filter floor, nhan dien toan bo"
        )
        return None
    except Exception as e:
        logger.error(f"Loi _get_valid_student_ids: {e}")
        return None


def _find_camera_by_id(camera_id_str: str):
    if not camera_id_str:
        return None
    cameras = camera_repo.get_all()
    # 1. Numeric ID (db_camera_id tu headless_processor)
    if camera_id_str.isdigit():
        cam = next((c for c in cameras if c.camera_id == int(camera_id_str)), None)
        if cam:
            return cam
    # 2. Resolve qua edge_status
    for dev_status in state_manager.get_all_edge_status().values():
        cam_status = dev_status.get("camera_status", {})
        if camera_id_str in cam_status and isinstance(cam_status[camera_id_str], dict):
            name = cam_status[camera_id_str].get("name", camera_id_str)
            cam = next((c for c in cameras if c.camera_name == name), None)
            if cam:
                return cam
    # 3. Match theo ten hoac RTSP URL
    return next(
        (c for c in cameras
         if c.camera_name == camera_id_str
         or c.rtsp_url == camera_id_str
         or (c.effective_rtsp_url and c.effective_rtsp_url == camera_id_str)),
        None
    )

@router.get("/api/students")
async def get_students(
    class_id: Optional[str] = Query(None),
    api_key: str = Security(verify_device_access),
):
    try:
        students = student_repo.get_by_class(class_id) if class_id else student_repo.get_all()
        return {
            "status": "success",
            "data": [
                {"student_id": s.student_id, "student_code": s.student_code,
                 "full_name": s.full_name, "class_name": s.class_name}
                for s in students
            ],
        }
    except Exception as e:
        logger.error(f"Loi /students: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/embeddings")
async def get_embeddings(
    camera_id: Optional[str] = Query(None),
    api_key: str = Security(verify_device_access)
):
    try:
        if embedding_service.size == 0:
            return {"status": "success", "count": 0, "students": []}

        valid_student_ids = None
        if camera_id:
            camera = _find_camera_by_id(camera_id)
            if camera:
                flr = getattr(camera, "floor", None)
                bld = getattr(camera, "device_group", None)
                if (bld is None or flr is None) and camera.area_id:
                    parts = camera.area_id.split("_", 1)
                    bld = parts[0].strip() if len(parts) > 0 else bld
                    flr = parts[1].strip() if len(parts) > 1 else flr
                valid_student_ids = _get_valid_student_ids(bld, str(flr)) if flr else None

        result = []
        all_embs = embedding_service.get_all_embeddings()
        for i in range(embedding_service.size):
            if valid_student_ids is not None and all_embs["student_ids"][i] not in valid_student_ids:
                continue
            emb_b64 = base64.b64encode(all_embs["embeddings"][i].tobytes()).decode("utf-8")
            result.append({
                "student_id": all_embs["student_ids"][i],
                "student_code": all_embs["student_codes"][i],
                "full_name": all_embs["full_names"][i],
                "class_id": all_embs["class_ids"][i],
                "class_name": all_embs["class_names"][i],
                "class_code": all_embs.get("class_codes", [""] * embedding_service.size)[i],
                "embedding_b64": emb_b64,
            })

        logger.info(f"Gui {len(result)}/{embedding_service.size} embeddings cho camera_id={camera_id}")
        return {"status": "success", "count": len(result), "students": result}
    except Exception as e:
        logger.error(f"Loi /embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/embeddings/version")
async def get_embedding_version(api_key: str = Security(verify_device_access)):
    ver, updated_at = state_manager.get_embedding_version()
    return {"embedding_version": ver, "updated_at": updated_at, "total": embedding_service.size}

@router.get("/api/sessions/active")
async def get_active_sessions(api_key: str = Security(verify_device_access)):
    try:
        all_sessions = session_repo.get_all(limit=50)
        active = [s for s in all_sessions if s.status == "ACTIVE"]
        return {
            "status": "success",
            "data": [
                {
                    "session_id": s.session_id, "class_name": s.class_name,
                    "subject_name": s.subject_name,
                    "session_date": str(s.session_date) if s.session_date else "",
                    "start_time": s.start_time.isoformat() if s.start_time else "",
                    "present_count": s.present_count,
                }
                for s in active
            ],
        }
    except Exception as e:
        logger.error(f"Loi /sessions/active: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/attendance", response_model=AttendanceResponse)
async def receive_attendance(
    payload: AttendancePayload,
    api_key: str = Security(verify_device_access),
):
    import numpy as np
    try:
        incoming_vector = np.array(payload.embedding, dtype=np.float32)
        if len(incoming_vector) != 512:
            raise HTTPException(status_code=400, detail="Vector khong hop le (can 512 chieu)")

        if payload.liveness_checked and payload.liveness_score < 0.80:
            attendance_logger.warning(
                f"[SPOOFING] Score: {payload.liveness_score:.3f} | Cam: {payload.camera_id}"
            )
            return AttendanceResponse(
                status="rejected",
                message=f"Phat hien gia mao (Liveness: {payload.liveness_score:.2f})",
            )

        if embedding_service.size == 0:
            return AttendanceResponse(status="ignored", message="CSDL trong")

        # Tim camera va filter hoc vien theo tang
        camera = None
        valid_student_ids = None
        if payload.camera_id:
            camera = _find_camera_by_id(payload.camera_id)
            if camera:
                flr = getattr(camera, "floor", None)
                bld = getattr(camera, "device_group", None)
                if (bld is None or flr is None) and camera.area_id:
                    parts = camera.area_id.split("_", 1)
                    bld = parts[0].strip() if len(parts) > 0 else bld
                    flr = parts[1].strip() if len(parts) > 1 else flr
                valid_student_ids = _get_valid_student_ids(bld, str(flr)) if flr else None
                count_str = str(len(valid_student_ids)) if valid_student_ids is not None else "all"
                logger.info(
                    f"Camera [{camera.camera_id}] floor={flr} group={bld} -> {count_str} SV kha dung"
                )
            else:
                logger.warning(f"Khong tim thay camera_id='{payload.camera_id}' trong DB")

        # Nhan dien
        best_score, best_idx = embedding_service.search(
            incoming_vector=incoming_vector,
            top_k=1,
            valid_ids=valid_student_ids
        )

        if best_idx != -1 and best_score >= ai_config.recognition_threshold:
            student_info = embedding_service.get_student_info(best_idx)
            student_id   = student_info["student_id"]
            student_code = student_info["student_code"]
            full_name    = student_info["full_name"]
            class_name   = student_info["class_name"]

            cmd_state = state_manager.get_command_state()
            session_id = cmd_state["session_id"]
            active_session = None

            if session_id:
                active_session = session_repo.get_by_id(session_id)
                if active_session and active_session.status != "ACTIVE":
                    active_session = None

            if not active_session:
                all_sessions = session_repo.get_all(limit=5)
                active_list = [s for s in all_sessions if s.status == "ACTIVE"]
                if active_list:
                    active_session = active_list[0]
                    state_manager.set_command(
                        "START", active_session.session_id,
                        active_session.class_id, cmd_state.get("target_camera")
                    )

            if not active_session:
                return AttendanceResponse(
                    status="no_session",
                    message="Khong co phien diem danh",
                    student_code=student_code,
                    full_name=full_name,
                    class_name=class_name,
                    similarity=best_score,
                )

            db_cam_id = camera.camera_id if camera else 1
            task_payload = {
                "session_id": active_session.session_id,
                "student_id": student_id,
                "student_code": student_code,
                "full_name": full_name,
                "class_name": class_name,
                "recognition_score": best_score,
                "camera_id": db_cam_id,
                "timestamp": datetime.now().isoformat(),
                "retry_count": 0
            }
            state_manager.redis.lpush("queue:attendance", json.dumps(task_payload))
            q_size = state_manager.redis.llen("queue:attendance")
            attendance_logger.info(
                f"Dua {full_name} vao Queue (Score: {best_score:.2f} | QSize: {q_size})"
            )
            return AttendanceResponse(
                status="queued",
                message="Da dua vao hang doi xu ly",
                student_code=student_code,
                full_name=full_name,
                class_name=class_name,
                similarity=best_score,
                session_id=active_session.session_id,
            )
        else:
            return AttendanceResponse(
                status="unknown",
                message=f"Khong nhan dien duoc (Score: {best_score:.2f})",
            )
    except Exception as e:
        logger.error(f"Loi /attendance: {e}")
        raise HTTPException(status_code=500, detail=str(e))
