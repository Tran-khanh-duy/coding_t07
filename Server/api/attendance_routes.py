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
    Lay danh sach student_id theo toa nha (building) va tang (floor).

    Return semantics (THEP):
    - None       : KHONG co filter → caller nhan dien TOAN BO.
    - []         : Co filter nhung KHONG co SV thoa man → Edge tra Nguoi La.
    - [id1, ...] : Danh sach SV hop le.

    Chien luoc Bidirectional Fuzzy Match (PA2):
      Thoat khoi rang buoc "DB phai chinh xac 100% voi tham so API".
      Match neu MOT TRONG HAI dieu kien sau la dung:
        (A) Exact  : UPPER(TRIM(db.building)) = UPPER(param)
                     VD: DB="KTX E4", param="KTX E4" ✓
        (B) DB ⊂ Param: param CHUA DB value
                     VD: DB="E4",     param="KTX E4" ✓  (E4 la substring cua KTX E4)
        (C) Param ⊂ DB: DB CHUA param value
                     VD: DB="KTX E4 Tang 4", param="KTX E4" ✓

      Floor luon dung TRIM+CAST de chong loi INT vs VARCHAR.
    """
    building_clean = (building or "").strip()
    floor_clean    = str(floor or "").strip()

    if not building_clean and not floor_clean:
        return None

    try:
        from database.connection import get_db
        db = get_db()

        # ── Bidirectional Fuzzy Match (1 query, 1 round-trip DB) ──────────────
        # MySQL khong co ILIKE, dung UPPER() de case-insensitive.
        # CONCAT('%', X, '%') thay cho f-string de tranh SQL injection.
        building_params: list = []
        building_clause: str = ""

        if building_clean:
            building_clause = """(
                UPPER(TRIM(hv.building)) = UPPER(%s)
                OR UPPER(%s) LIKE CONCAT('%%', UPPER(TRIM(hv.building)), '%%')
                OR UPPER(TRIM(hv.building)) LIKE CONCAT('%%', UPPER(%s), '%%')
            )"""
            # 3 params tuong ung 3 %s trong clause tren
            building_params = [building_clean, building_clean, building_clean]

        floor_params: list = []
        floor_clause: str = ""

        if floor_clean:
            floor_clause = "TRIM(CAST(hv.floor AS CHAR)) = %s"
            floor_params = [floor_clean]

        # Ghep WHERE clause
        clauses = [c for c in [building_clause, floor_clause] if c]
        if not clauses:
            return None  # Khong co filter

        sql = (
            "SELECT hv.id FROM hocvien hv WHERE "
            + " AND ".join(clauses)
        )
        all_params = tuple(building_params + floor_params)

        rows = db.execute(sql, all_params)

        if rows:
            ids = [int(r[0]) for r in rows]
            logger.info(
                f"[FILTER OK] [{building_clean} T{floor_clean}]: "
                f"{len(ids)} SV hop le (fuzzy match)"
            )
            return ids

        # ── Khong match — in DISTINCT thuc te de debug ngay ──────────────────
        try:
            # FIX only_full_group_by: XOA ORDER BY khoi SQL,
            # fetch het roi sort bang Python de tranh vi pham strict mode MySQL.
            raw_blds = db.execute(
                "SELECT TRIM(building) AS b, COUNT(*) AS c "
                "FROM hocvien WHERE building IS NOT NULL "
                "GROUP BY TRIM(building)"
                # KHONG co ORDER BY — MySQL strict mode se loi neu ORDER BY
                # dung expression khong nam trong GROUP BY
            )
            # Sort Python: theo ten building
            raw_blds = sorted(raw_blds, key=lambda r: (r[0] or "").lower())
            bld_list = [f'"{r[0]}" ({r[1]} SV)' for r in raw_blds] or ["(trong)"]

            raw_flrs = db.execute(
                "SELECT TRIM(CAST(floor AS CHAR)) AS f, COUNT(*) AS c "
                "FROM hocvien "
                "WHERE floor IS NOT NULL "
                "AND (UPPER(TRIM(building)) = UPPER(%s) "
                "     OR UPPER(%s) LIKE CONCAT('%%', UPPER(TRIM(building)), '%%') "
                "     OR UPPER(TRIM(building)) LIKE CONCAT('%%', UPPER(%s), '%%')) "
                "GROUP BY TRIM(CAST(floor AS CHAR))"
                # KHONG co ORDER BY — sort bang Python ben duoi
                ,
                (building_clean, building_clean, building_clean)
            )
            # Sort Python: so nguyen truoc, chuoi sau
            def _floor_key(r):
                try:
                    return (0, int(r[0]))
                except (TypeError, ValueError):
                    return (1, str(r[0] or ""))
            raw_flrs = sorted(raw_flrs, key=_floor_key)
            flr_list = [f'"{r[0]}"' for r in raw_flrs] or ["(khong co tang nao match building nay)"]

        except Exception as diag_err:
            logger.debug(f"Khong lay duoc DISTINCT de diagnostic: {diag_err}")
            bld_list = ["(loi truy van diagnostic)"]
            flr_list = ["(loi truy van diagnostic)"]

        logger.warning(
            f"[STRICT FILTER] Khong tim thay SV nao sau Fuzzy Match:\n"
            f"  Tham so    : building='{building_clean}' floor='{floor_clean}'\n"
            f"  DB building: {', '.join(bld_list)}\n"
            f"  DB floor   : {', '.join(flr_list)}\n"
            f"  Logic da thu: (A) exact, (B) DB⊂param, (C) param⊂DB\n"
            f"  => Khong co gi khop. Kiem tra lai du lieu nhap vao DB.\n"
            f"  => Chay: cd Server && python check_db_data.py "
            f"--building \"{building_clean}\" --floor \"{floor_clean}\""
        )
        return []

    except Exception as exc:
        logger.exception(
            f"Loi _get_valid_student_ids("
            f"building={building_clean!r}, floor={floor_clean!r}): {exc}"
        )
        return []



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
                valid_student_ids = _get_valid_student_ids(bld, str(flr)) if flr is not None else None

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

        if payload.liveness_checked and payload.liveness_score < getattr(ai_config, "liveness_threshold", 0.80):
            attendance_logger.warning(
                f"[SPOOFING] Score: {payload.liveness_score:.3f} "
                f"(threshold={getattr(ai_config, 'liveness_threshold', 0.80)}) "
                f"| Cam: {payload.camera_id}"
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
                # Fallback: lay tu area_id neu device_group/floor chua duoc set
                if (bld is None or flr is None) and camera.area_id:
                    parts = camera.area_id.split("_", 1)
                    bld = parts[0].strip() if len(parts) > 0 else bld
                    flr = parts[1].strip() if len(parts) > 1 else flr

                # Chi filter neu CO CA building lan floor:
                # - Neu chi co floor ma khong biet building → nguy hiem (match SV sai toa).
                # - Neu chi co building ma khong biet floor → filter theo toa (an toan).
                if flr is not None and bld is not None:
                    valid_student_ids = _get_valid_student_ids(bld, str(flr))
                elif bld is not None:
                    valid_student_ids = _get_valid_student_ids(bld, "")
                else:
                    valid_student_ids = None  # Khong du thong tin → nhan dien toan bo

                count_str = (
                    str(len(valid_student_ids)) if valid_student_ids
                    else "0 (filter active, no match)" if valid_student_ids is not None
                    else "all (no filter)"
                )
                logger.info(
                    f"Camera [{camera.camera_id}] bld='{bld}' floor='{flr}' "
                    f"-> valid_ids={count_str}"
                )
            else:
                logger.warning(
                    f"Khong tim thay camera_id='{payload.camera_id}' trong DB "
                    f"— nhan dien TOAN BO (co the gay False Positive)."
                )

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
