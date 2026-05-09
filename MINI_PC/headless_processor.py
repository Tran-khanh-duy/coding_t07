"""
services/headless_processor.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Headless AI Processor for Multi-Camera (Edge Box)
- Multi-threaded: Mỗi camera 1 thread xử lý riêng.
- Shared AI models và shared Edge Client.
- Tích hợp vẽ Bounding Box có Tracking nội suy.
- Hỗ trợ Render Font Tiếng Việt chuẩn xác qua Pillow (Font To, Đậm).
- Chế độ Idle: Nằm im chờ lệnh START từ Server.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os
import time
import cv2
import numpy as np
import threading
from datetime import datetime
import requests
import base64
import psutil
from loguru import logger
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from config import edge_config, ai_config, anti_spoof_config, camera_config
# pyrefly: ignore [missing-import]
from services.face_engine import face_engine
try:
    # pyrefly: ignore [missing-import]
    from services.anti_spoof_service import anti_spoof_service  
    ANTI_SPOOF_AVAILABLE = anti_spoof_service.available
    if ANTI_SPOOF_AVAILABLE:
        logger.success("🚀 [Edge] Anti-Spoofing đã sẵn sàng!")
    else:
        logger.warning("⚠️ [Edge] Anti-Spoofing không khả dụng.")
except Exception as e:
    logger.warning(f"[Edge] Anti-Spoofing import thất bại: {e}")
    anti_spoof_service = None
    ANTI_SPOOF_AVAILABLE = False

from edge_client import edge_client
from local_cache.attendance_cache import attendance_cache

# Registry toàn cục để theo dõi các cổng phần cứng đang bận
os.environ["OPENCV_VIDEOIO_PRIORITY_OBSENSOR"] = "0" 
# ÉP FFMPEG CHẠY CHẾ ĐỘ LOW LATENCY CỰC ĐOAN
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|rtsp_flags;nobuffer|probesize;32|analyzeduration;0|fflags;nobuffer|flags;low_delay"

ACTIVE_SOURCES = set()
SOURCES_LOCK = threading.Lock()

import queue

class CameraWorker:
    """
    Luồng xử lý Camera theo kiến trúc Pipeline (Non-blocking):
    Thread 1 (Capture) -> Thread 2 (Detection) -> Thread 3 (Recognition) -> Thread 4 (API Sender)
    """
    
    def __init__(self, camera_id: str, source: str, db_camera_id: int = None):
        self.camera_id = camera_id        # VD: "CAM_01" — dùng cho hiển thị/live view
        self.db_camera_id = db_camera_id  # VD: 1 (int) — dùng để gửi lên attendance API
        self.source = source
        self._running = False
        self._stop_event = threading.Event()
        
        # Trạng thái điều khiển
        self._active = False 
        self._is_previewing = False 
        self._attendance_enabled = False 
        
        # Buffers cho Live View
        self._latest_frame = None
        self._last_known_faces = []
        self._frame_lock = threading.Lock()
        
        # Trí nhớ ngắn hạn cho chống giả mạo / nhận diện liên tiếp
        self._real_face_history = {}
        self._spoof_log_cache = {}

        # 🚀 Pipeline Queues (Kích thước nhỏ để rơi frame cũ, đảm bảo realtime)
        self.detect_queue = queue.Queue(maxsize=2)
        self.recognize_queue = queue.Queue(maxsize=2)
        self.api_queue = queue.Queue(maxsize=10)

        # Threads
        self.threads = []
        self._capture_frame_count = 0
        
        # Watchdog Health Metrics
        self.last_capture_time = time.time()
        self.last_ai_time = time.time()

    def set_active(self, active: bool):
        self._active = active

    def set_previewing(self, previewing: bool):
        self._is_previewing = previewing

    def set_attendance_enabled(self, enabled: bool):
        self._attendance_enabled = enabled
        if enabled:
            # Xoá rác từ phiên cũ để tránh ghi nhận lệch phiên
            self._real_face_history.clear()
            self._spoof_log_cache.clear()
            
            # Reset cooldown trên edge client
            from edge_client import edge_client
            edge_client.reset_cooldown()
            
            # Xoá sạch các hàng đợi để tránh "bóng ma" từ phiên trước
            import queue
            for q in [self.detect_queue, self.recognize_queue, self.api_queue]:
                try:
                    while True: q.get_nowait()
                except queue.Empty:
                    pass

    def start(self):
        if self._running: return
        self._running = True
        self._stop_event.clear()
        
        # Tách làm 5 Luồng: 4 Luồng Pipeline + 1 Luồng Live View
        self.threads = [
            threading.Thread(target=self._capture_loop, name=f"Cap-{self.camera_id}", daemon=True),
            threading.Thread(target=self._detect_loop, name=f"Det-{self.camera_id}", daemon=True),
            threading.Thread(target=self._recognize_loop, name=f"Rec-{self.camera_id}", daemon=True),
            threading.Thread(target=self._api_loop, name=f"Api-{self.camera_id}", daemon=True),
            threading.Thread(target=self._live_loop, name=f"Liv-{self.camera_id}", daemon=True)
        ]
        
        for t in self.threads:
            t.start()
        
        # Khởi tạo cache độc lập cho camera này (dùng db_camera_id để filter floor trên Server)
        threading.Thread(
            target=lambda: edge_client.pull_embeddings(self.camera_id, db_camera_id=self.db_camera_id),
            daemon=True
        ).start()
        
        logger.info(f"🚀 CameraWorker {self.camera_id} khởi động kiến trúc Multi-thread Pipeline (4 luồng).")

    def _capture_loop(self):
        """THREAD 1: Camera Capture -> Detect Queue"""
        cap = None
        while not self._stop_event.is_set():
            if not self._active and not self._is_previewing:
                if cap:
                    cap.release()
                    cap = None
                    edge_client.update_active_status(self.camera_id, False)
                    with SOURCES_LOCK:
                        if self.source in ACTIVE_SOURCES: ACTIVE_SOURCES.remove(self.source)
                time.sleep(0.5)
                continue

            if cap is None or not cap.isOpened():
                cap = self._open_camera_backend()
                if cap is None:
                    time.sleep(2)
                    continue

            edge_client.update_active_status(self.camera_id, True)
            ret, frame = cap.read()
            self.last_capture_time = time.time() # Update Watchdog
            if not ret:
                logger.warning(f"⚠️ Camera {self.camera_id}: Mất tín hiệu, đang thử lại...")
                cap.release()
                cap = None
                edge_client.update_active_status(self.camera_id, False)
                time.sleep(1)
                continue

            # Update Live View Buffer (Hiển thị 30 FPS mượt mà)
            with self._frame_lock:
                self._latest_frame = frame

            # Đẩy vào Detect Queue với frame_skip (Giảm tải CPU/GPU)
            self._capture_frame_count += 1
            if self._capture_frame_count % edge_config.frame_skip != 0:
                continue

            try:
                if self.detect_queue.full():
                    self.detect_queue.get_nowait()
                capture_time = datetime.now().isoformat()
                self.detect_queue.put_nowait((frame.copy(), capture_time))
            except queue.Empty:
                pass
            except queue.Full:
                pass

        if cap: cap.release()

    def _detect_loop(self):
        """THREAD 2: Detect Queue -> Detections -> Recognize Queue"""
        while not self._stop_event.is_set():
            try:
                frame, capture_time = self.detect_queue.get(timeout=0.5)
            except queue.Empty:
                continue
                
            if not self._active or not self._attendance_enabled:
                self._last_known_faces = []
                continue

            try:
                detected = face_engine.detect_faces(frame)
                if detected:
                    # Gửi sang luồng Recognize
                    try:
                        if self.recognize_queue.full():
                            self.recognize_queue.get_nowait()
                        self.recognize_queue.put_nowait((frame, detected, capture_time))
                    except queue.Empty:
                        pass
                    except queue.Full:
                        pass
                else:
                    self._last_known_faces = []
            except Exception as e:
                logger.error(f"❌ Detect Loop Error [{self.camera_id}]: {e}")

    def _recognize_loop(self):
        """THREAD 3: Recognize Queue -> Recognition -> API Queue"""
        while not self._stop_event.is_set():
            try:
                frame, detected, capture_time = self.recognize_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                cache = edge_client.get_cache(self.camera_id)
                results = face_engine.recognize_batch(detected, cache)
                
                new_known_faces = []
                for i, res in enumerate(results):
                    # Xử lý Anti-spoofing trực tiếp trong Thread 3 (GPU bound)
                    is_real = True
                    spoof_score = 1.0
                    
                    if res.recognized:
                        # Fast-Path Cooldown: Chi quet chong gia mao neu hoc sinh CHUA diem danh
                        remaining = edge_client.check_cooldown(res.student_id, self.camera_id)
                        if remaining <= 0:
                            if self._attendance_enabled and ANTI_SPOOF_AVAILABLE and anti_spoof_service:
                                # Anti-Spoof kha dung: ket qua thuc te
                                is_real, spoof_score = anti_spoof_service.is_real(frame, res.bbox)
                                res.is_real    = is_real
                                res.spoof_score = spoof_score
                            else:
                                # Anti-Spoof KHONG kha dung: mac dinh cho phep qua
                                # ROOT CAUSE FIX: neu khong set is_real=True o day,
                                # res.is_real van la None -> _api_loop check 'res.is_real'
                                # se False -> diem danh bi block hoan toan!
                                res.is_real    = True
                                res.spoof_score = 1.0
                        else:
                            # Da diem danh xong -> bo qua Anti-Spoofing nang ne
                            res.is_real    = True
                            res.spoof_score = 1.0
                    
                    color_val = "unknown"
                    if res.recognized:
                        color_val = "success" if res.is_real else "danger"
                    
                    new_known_faces.append({
                        "bbox": detected[i].bbox,
                        "name": res.display_name if res.recognized else "Unknown",
                        "color_type": color_val
                    })

                    if res.recognized:
                        payload = {
                            "result": res,
                            "embedding": detected[i].embedding,
                            "camera_id": self.camera_id,
                            "timestamp": capture_time
                        }
                        try:
                            self.api_queue.put_nowait(payload)
                        except queue.Full:
                            logger.warning("⚠️ API Queue đầy, bỏ qua nhận diện hiện tại.")

                self._last_known_faces = new_known_faces
                self.last_ai_time = time.time() # Update Watchdog
            except Exception as e:
                logger.error(f"❌ Recognize Loop Error [{self.camera_id}]: {e}")

    def _api_loop(self):
        """THREAD 4: API Queue -> HTTP Request (I/O Bound)"""
        while not self._stop_event.is_set():
            try:
                payload = self.api_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            res = payload["result"]
            embedding = payload["embedding"]
            cam_id = payload["camera_id"]
            capture_time = payload["timestamp"]

            try:
                if self._attendance_enabled and res.is_real:
                    current_count = self._real_face_history.get(res.student_id, 0)
                    self._real_face_history[res.student_id] = current_count + 1

                    needed = getattr(edge_config, "accumulation_frames", 3)
                    accumulated = self._real_face_history[res.student_id]
                    logger.debug(
                        f"[API-LOOP] [{self.camera_id}] '{res.display_name}' "
                        f"tich luy {accumulated}/{needed} frame(s)"
                    )

                    if accumulated >= needed:
                        remaining = edge_client.check_cooldown(res.student_id, cam_id)
                        if remaining <= 0:
                            # 1. OFFLINE-FIRST: Luu vao SQLite ngay lap tuc
                            record_id = attendance_cache.save_pending(
                                camera_id=cam_id,
                                embedding=embedding,
                                liveness_score=res.spoof_score,
                                liveness_checked=ANTI_SPOOF_AVAILABLE,
                                timestamp=capture_time
                            )

                            # 2. Gui len Server API
                            api_cam_id = self.db_camera_id if self.db_camera_id else cam_id
                            logger.info(
                                f"[API-LOOP] [{self.camera_id}] Gui diem danh: "
                                f"'{res.display_name}' | cam_api={api_cam_id} "
                                f"| spoof={res.spoof_score:.2f} "
                                f"| anti_spoof_active={ANTI_SPOOF_AVAILABLE}"
                            )
                            result = edge_client.send_attendance_raw(
                                embedding=embedding,
                                camera_id=str(api_cam_id),
                                liveness_score=res.spoof_score,
                                liveness_checked=ANTI_SPOOF_AVAILABLE,
                                timestamp=capture_time
                            )

                            status = result.get("status", "unknown")
                            if status in ("success", "ignored", "queued"):
                                attendance_cache.remove_pending(record_id)
                                logger.success(
                                    f"[API-LOOP] [{self.camera_id}] Server chap nhan: "
                                    f"'{res.display_name}' | status='{status}'"
                                )
                            else:
                                attendance_cache.mark_failed(record_id, result.get("message", "API Error"))
                                logger.error(
                                    f"[API-LOOP] [{self.camera_id}] Server TU CHOI: "
                                    f"'{res.display_name}' | status='{status}' "
                                    f"| message='{result.get('message', '')}'"
                                )

                            edge_client.set_cooldown(res.student_id, cam_id)
                            self._real_face_history[res.student_id] = 0
                        else:
                            logger.debug(
                                f"[API-LOOP] [{self.camera_id}] '{res.display_name}' "
                                f"dang trong COOLDOWN con {remaining:.0f}s"
                            )
                elif self._attendance_enabled and not res.is_real:
                    self._log_spoof(res)
                    self._real_face_history[res.student_id] = 0
                elif self._attendance_enabled and res.is_real is None:
                    # Guard: is_real chua duoc set (bug) - log de phat hien
                    logger.warning(
                        f"[API-LOOP] [{self.camera_id}] BUG: res.is_real=None cho "
                        f"'{res.display_name}' — diem danh bi bo qua! "
                        f"Kiem tra _recognize_loop anti-spoof path."
                    )
            except Exception as api_err:
                logger.exception(f"[API-LOOP] [{self.camera_id}] CRITICAL exception: {api_err}")

            finally:
                self.api_queue.task_done()

    def _live_loop(self):
        """THREAD 5: Cập nhật Live View cho Server"""
        last_upload = 0
        while not self._stop_event.is_set():
            if not self._is_previewing:
                time.sleep(0.5)
                continue
            
            now = time.time()
            if now - last_upload < 0.05: # Giới hạn 20 FPS cho Live View để mượt hơn
                time.sleep(0.01)
                continue

            frame = None
            with self._frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()
            
            if frame is None:
                time.sleep(0.05)
                continue

            last_upload = now
            try:
                # Lấy kết quả AI mới nhất để vẽ (nội suy)
                known_faces = self._last_known_faces
                dets_payload = []
                
                # Vẽ và chuẩn bị payload
                if known_faces:
                    for face in known_faces:
                        x1, y1, x2, y2 = face['bbox']
                        name = face['name']
                        color_type = face['color_type']
                        dets_payload.append([int(x1), int(y1), int(x2), int(y2), name, color_type])
                
                # Thực hiện vẽ và upload (vẫn dùng thread ephemeral để không block loop này)
                self._upload_frame_async(frame, dets_payload)
            except Exception as e:
                logger.error(f"❌ Live Loop Error [{self.camera_id}]: {e}")

    def _open_camera_backend(self):
        source = self.source
        with SOURCES_LOCK:
            if source in ACTIVE_SOURCES: return None
            ACTIVE_SOURCES.add(source)
        try:
            try:
                cam_idx = int(source)
                cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            except:
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
            if cap and cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            else:
                with SOURCES_LOCK: ACTIVE_SOURCES.remove(source)
                return None
        except:
            with SOURCES_LOCK: 
                if source in ACTIVE_SOURCES: ACTIVE_SOURCES.remove(source)
            return None

    def _upload_frame_async(self, frame, dets=None):
        # Giữ nguyên logic vẽ và upload cũ nhưng bọc trong try/except
        cam_id = self.camera_id
        def task():
            try:
                frame_copy = frame # Đã copy ở loop ngoài rồi
                if dets:
                    for det in dets:
                        x1, y1, x2, y2, name, color_type = det
                        color_cv = (0, 255, 0) if color_type == "success" else (0, 0, 255) if color_type == "danger" else (0, 255, 255)
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color_cv, 4)

                    # Vẽ Tiếng Việt
                    frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    draw = ImageDraw.Draw(pil_img)
                    font_size = 52
                    try: font = ImageFont.truetype("arialbd.ttf", font_size)
                    except: 
                        try: font = ImageFont.truetype("arial.ttf", font_size)
                        except: font = ImageFont.load_default()

                    for det in dets:
                        x1, y1, x2, y2, name, color_type = det
                        color_pil = (0, 255, 0) if color_type == "success" else (255, 0, 0) if color_type == "danger" else (255, 255, 0)
                        if hasattr(font, 'getbbox'):
                            bbox = font.getbbox(name); tw = bbox[2]-bbox[0]; th = bbox[3]-bbox[1]
                        else: tw, th = draw.textsize(name, font=font)
                        bg_x1, bg_y1 = max(0, int(x1)), max(0, int(y1) - int(th) - 20)
                        draw.rectangle([(bg_x1, bg_y1), (int(bg_x1 + tw + 20), int(bg_y1 + th + 20))], fill=(0, 0, 0))
                        draw.text((bg_x1 + 10, bg_y1 + 4), name, font=font, fill=color_pil)
                    frame_copy = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                h, w = frame_copy.shape[:2]
                sw = 640; sh = int(h * (sw / w))
                small = cv2.resize(frame_copy, (sw, sh))
                _, buffer = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
                img_b64 = base64.b64encode(buffer).decode("utf-8")
                
                url = f"{edge_client.server_url}/api/system/frame"
                payload = {"image_b64": img_b64, "camera_id": cam_id, "detections": dets or []}
                headers = {"X-DEVICE-TOKEN": edge_client._headers()["X-DEVICE-TOKEN"]}
                requests.post(url, json=payload, headers=headers, timeout=5)
            except: pass
        
        threading.Thread(target=task, daemon=True).start()

    def _log_spoof(self, res):
        now = time.time()
        # Chỉ log spoof cho học viên đã nhận diện được, với cooldown 30 giây
        if not res.recognized:
            return
        key = res.student_id
        if now - self._spoof_log_cache.get(key, 0) > 30.0:
            logger.warning(f"🚫 [{self.camera_id}] Phát hiện GIẢ MẠO: {res.display_name} | Score: {res.spoof_score:.1%}")
            self._spoof_log_cache[key] = now

class HeadlessProcessor:
    def __init__(self):
        self._workers = {}
        self._running = False
        self._current_command = "STOP" # Trạng thái ban đầu luôn là STOP
        self._target_camera_view = None
        self._last_embed_refresh = 0.0

    def start(self):
        if self._running: return
        logger.info("🚀 Headless Processor (Multi-Cam) đang khởi động...")
        if not face_engine.load_model():
            logger.error("❌ Không thể nạp model AI.")
            return

        cam_list = getattr(edge_config, "camera_list", [])
        if not cam_list:
            logger.warning("⚠️ camera_list rỗng, thử pull lại từ Server...")
            from edge_client import edge_client as _ec
            _ec.pull_camera_list()
            cam_list = getattr(edge_config, "camera_list", [])
        if not cam_list:
            logger.error("❌ Không có camera nào. Kiểm tra: (1) Server đang chạy, (2) Bảng Cameras có dữ liệu trong DB.")
            return

        # Doc tu config: neu EDGE_AUTO_START=true, camera tu dong bat ngay khi khoi dong
        is_auto = getattr(edge_config, 'auto_start', False)
        if is_auto:
            logger.info("🟢 Auto-start = True: Camera se bat ngay khi khoi dong.")
        else:
            logger.info("⏸️ Mini PC da san sang va dang CHO LENH. Hay bam Bat dau tren Server...")

        
        for cam in cam_list:
            cid = cam["id"]             # "CAM_01"
            src = cam["source"]         # rtsp://...
            db_id = cam.get("camera_id")  # 1 (int từ DB)
            worker = CameraWorker(cid, src, db_camera_id=db_id)
            worker.set_active(is_auto)
            worker.set_attendance_enabled(is_auto)
            self._workers[cid] = worker
            worker.start()
            time.sleep(0.2)

        self._running = True
        
        # Bật Watchdog
        threading.Thread(target=self._watchdog_loop, name="Watchdog", daemon=True).start()
        
        self._run_control_loop()

    def _run_control_loop(self):
        last_command_check = 0
        while self._running:
            now = time.time()
            
            # Cứ mỗi 1 giây, Mini PC sẽ gọi API lên Server để "hỏi" xem có lệnh mới không
            if now - last_command_check >= 1.0:
                last_command_check = now
                try:
                    cmd_data = edge_client.get_system_command_raw()
                    new_cmd = cmd_data.get("command", "STOP")
                    actual_target = cmd_data.get("target_camera")

                    # NẾU PHÁT HIỆN LỆNH MỚI TỪ SERVER
                    if new_cmd != self._current_command:
                        if new_cmd == "RETRY_CAMERA" and actual_target:
                            logger.warning(f"🔄 NHẬN LỆNH [RETRY]: Khởi động lại Camera {actual_target}")
                            target_key = None
                            for cid, worker in self._workers.items():
                                if str(cid).upper() == str(actual_target).upper() or str(worker.source).upper() == str(actual_target).upper():
                                    target_key = cid
                                    worker.set_active(False)
                                    worker._stop_event.set()
                                    break
                            
                            if target_key:
                                old_worker = self._workers[target_key]
                                new_worker = CameraWorker(
                                    camera_id=target_key,
                                    source=old_worker.source,
                                    db_camera_id=old_worker.db_camera_id
                                )
                                new_worker.set_active(True)
                                new_worker.set_attendance_enabled(True)
                                new_worker.set_previewing(True)
                                self._workers[target_key] = new_worker
                                new_worker.start()
                                
                            # Reset lệnh retry ngay để không lặp lại
                            try: requests.post(f"{edge_client.server_url}/api/system/command", json={"command": "START"}, headers=edge_client._headers(), timeout=1)
                            except: pass
                        else:
                            self._current_command = new_cmd
                            is_start = (new_cmd == "START")
                            
                            # In log thông báo trạng thái
                            if is_start:
                                logger.info("🟢 NHẬN LỆNH [START]: Đánh thức Camera, bắt đầu điểm danh!")
                                face_engine.load_model()
                                edge_client.reset_cooldown()
                            else:
                                logger.info("🔴 NHẬN LỆNH [STOP]: Tạm dừng điểm danh, giải phóng Camera.")
                                # giúp điểm danh khởi động lại ngay lập tức (không bị delay load vài giây).
                                # face_engine.unload_model()

                            # Đẩy lệnh xuống điều khiển tất cả các luồng camera
                            for worker in self._workers.values():
                                worker.set_active(is_start)
                                worker.set_attendance_enabled(is_start)

                    # Cập nhật xem Server có đang muốn xem trước (Preview) camera nào không
                    self._target_camera_view = actual_target
                    
                    # [NEW] Khởi tạo Worker cho các camera mới từ DB (dựa trên all_cameras trả về)
                    all_cams = cmd_data.get("all_cameras", [])
                    
                    # Fallback: đảm bảo target_camera cũng được khởi tạo
                    if actual_target and isinstance(actual_target, str):
                        if actual_target not in all_cams:
                            all_cams.append(actual_target)
                            
                    for cam_url in all_cams:
                        if isinstance(cam_url, str):
                            source_exists = any(w.source == cam_url for w in self._workers.values())
                            if cam_url not in self._workers and not source_exists:
                                # Tìm camera_id tương ứng trong DB list
                                cam_info = next(
                                    (c for c in edge_config.camera_list if c.get("source") == cam_url),
                                    None
                                )
                                cam_id = cam_info["id"] if cam_info else cam_url
                                cam_name = cam_info["name"] if cam_info else cam_url
                                db_id = cam_info.get("camera_id") if cam_info else None
                                logger.info(f"✨ Khởi tạo on-the-fly Worker: [{cam_id}] {cam_name}")
                                new_worker = CameraWorker(camera_id=cam_id, source=cam_url, db_camera_id=db_id)
                                is_sys_start = (self._current_command == "START")
                                new_worker.set_active(is_sys_start) 
                                new_worker.set_attendance_enabled(is_sys_start) 
                                self._workers[cam_id] = new_worker
                                new_worker.start()

                    for cid, worker in self._workers.items():
                        is_match = (cid.upper() == str(actual_target).upper() or str(actual_target).upper() == str(worker.source).upper())
                        worker.set_previewing(is_match)
                        
                    # [NEW] Tự động khởi tạo Worker cho các IP Camera được phát hiện tự động
                    for rtsp_url in edge_client._discovered_rtsp:
                        source_exists = any(w.source == rtsp_url for w in self._workers.values())
                        if rtsp_url not in self._workers and not source_exists:
                            logger.info(f"✨ Tự động nhận diện IP Camera mới: {rtsp_url}")
                            new_worker = CameraWorker(camera_id=rtsp_url, source=rtsp_url)
                            is_sys_start = (self._current_command == "START")
                            new_worker.set_active(is_sys_start)
                            # is_sys_start sẽ được truyền vào set_attendance_enabled, 
                            # bên trong method này nó đã tự lọc chỉ chạy cho rtsp:// rồi
                            new_worker.set_attendance_enabled(is_sys_start)
                            self._workers[rtsp_url] = new_worker
                            new_worker.start()
                            
                except Exception as e:
                    logger.error(f"Lỗi poll lệnh server: {e}")

            if now - self._last_embed_refresh >= 600:
                self._last_embed_refresh = now
                edge_client.pull_embeddings()

            time.sleep(0.1)

    def _watchdog_loop(self):
        """THREAD 6: Auto Recovery Watchdog"""
        logger.info("🛡️ Watchdog đã khởi động, giám sát RAM, FPS và Timeout.")
        while self._running:
            time.sleep(5.0)
            now = time.time()
            
            try:
                # 1. Kiểm tra RAM
                ram_percent = psutil.virtual_memory().percent
                if ram_percent > 90.0:
                    logger.critical(f"🔥 BÁO ĐỘNG: RAM quá tải ({ram_percent}%) - Kích hoạt Garbage Collector!")
                    import gc
                    gc.collect()

                # 2. Kiểm tra Health từng Worker
                workers_to_restart = []
                for cid, worker in list(self._workers.items()):
                    if not worker._active and not worker._is_previewing:
                        # Update time liên tục nếu đang dừng để tránh bị tính là timeout
                        worker.last_capture_time = now
                        worker.last_ai_time = now
                        continue
                    
                    # Capture Timeout (>15s không có frame mới)
                    if now - worker.last_capture_time > 15.0:
                        logger.error(f"💀 Watchdog: Camera {cid} bị treo Capture (>15s). Lên lịch Restart...")
                        workers_to_restart.append(cid)
                        continue
                    
                    # Inference Timeout (>20s không xử lý xong nhưng hàng đợi vẫn còn ảnh)
                    if now - worker.last_ai_time > 20.0 and not worker.recognize_queue.empty():
                        logger.error(f"💀 Watchdog: Camera {cid} bị treo AI Inference (>20s). Lên lịch Restart...")
                        workers_to_restart.append(cid)
                        continue

                # 3. Tự động Restart
                for cid in workers_to_restart:
                    self._restart_worker(cid)
                    
            except Exception as e:
                logger.error(f"Lỗi Watchdog: {e}")

    def _restart_worker(self, cid):
        logger.warning(f"🔄 [AUTO RECOVERY] Đang khởi động lại Camera: {cid}")
        old_worker = self._workers.get(cid)
        if not old_worker: return
        
        # Dừng worker cũ
        old_worker.set_active(False)
        old_worker._stop_event.set()
        
        # Khởi tạo lại
        new_worker = CameraWorker(camera_id=cid, source=old_worker.source)
        is_sys_start = (self._current_command == "START")
        new_worker.set_active(is_sys_start)
        new_worker.set_attendance_enabled(is_sys_start)
        new_worker.set_previewing(old_worker._is_previewing)
        
        self._workers[cid] = new_worker
        new_worker.start()
        logger.success(f"✅ [AUTO RECOVERY] Đã Restart thành công Camera Worker: {cid}")

    def stop(self):
        self._running = False
        for worker in self._workers.values():
            worker._stop_event.set()
        
        # Đợi các worker thread dừng hẳn
        for worker in self._workers.values():
            for t in worker.threads:
                if t.is_alive():
                    t.join(timeout=1.0)

headless_processor = HeadlessProcessor()