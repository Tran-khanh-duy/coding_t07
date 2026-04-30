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
import requests
import base64
from loguru import logger
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from config import edge_config, ai_config, anti_spoof_config, camera_config
from services.face_engine import face_engine
try:
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

# Registry toàn cục để theo dõi các cổng phần cứng đang bận
os.environ["OPENCV_VIDEOIO_PRIORITY_OBSENSOR"] = "0" 
# ÉP FFMPEG CHẠY CHẾ ĐỘ LOW LATENCY CỰC ĐOAN
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|rtsp_flags;nobuffer|probesize;32|analyzeduration;0|fflags;nobuffer|flags;low_delay"

ACTIVE_SOURCES = set()
SOURCES_LOCK = threading.Lock()

class CameraWorker:
    """Luồng xử lý cho một camera cụ thể - Đã tách riêng Capture, AI và LiveView."""
    
    def __init__(self, camera_id: str, source: str):
        self.camera_id = camera_id
        self.source = source
        self._running = False
        
        # Trạng thái điều khiển
        self._active = False 
        self._is_previewing = False 
        self._attendance_enabled = False 
        
        # Buffer dữ liệu dùng chung giữa các thread
        self._latest_frame = None
        self._latest_dets = []      # Kết quả AI mới nhất [x1, y1, x2, y2, name, color_type]
        self._frame_lock = threading.Lock()
        
        # "Trí nhớ ngắn hạn" cho tracking nội suy và xác thực
        self._last_known_faces = []
        self._real_face_history = {} # {student_id: count_consecutive_real}
        self._spoof_log_cache = {}

        # Threads
        self._capture_thread = None
        self._ai_thread = None
        self._live_thread = None

    def set_active(self, active: bool):
        self._active = active

    def set_previewing(self, previewing: bool):
        self._is_previewing = previewing

    def set_attendance_enabled(self, enabled: bool):
        self._attendance_enabled = enabled

    def start(self):
        if self._running: return
        self._running = True
        
        # 1. Luồng Capture: Chỉ đọc ảnh từ Camera và đẩy vào Buffer
        self._capture_thread = threading.Thread(target=self._capture_loop, name=f"Cap-{self.camera_id}", daemon=True)
        self._capture_thread.start()
        
        # 2. Luồng AI: Lấy ảnh từ Buffer và chạy Detection/Recognition
        self._ai_thread = threading.Thread(target=self._ai_loop, name=f"AI-{self.camera_id}", daemon=True)
        self._ai_thread.start()
        
        # 3. Luồng LiveView: Lấy ảnh + Kết quả AI và upload lên Server
        self._live_thread = threading.Thread(target=self._live_loop, name=f"Live-{self.camera_id}", daemon=True)
        self._live_thread.start()
        
        # Khởi tạo cache độc lập cho camera này
        threading.Thread(target=lambda: edge_client.pull_embeddings(self.camera_id), daemon=True).start()
        
        logger.info(f"🚀 CameraWorker {self.camera_id} đã khởi động với 3 luồng riêng biệt.")

    def _capture_loop(self):
        """Luồng đọc Camera: Phải chạy liên tục để không bị trễ buffer."""
        cap = None
        while self._running:
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
            if not ret:
                logger.warning(f"⚠️ Camera {self.camera_id}: Mất tín hiệu, đang thử lại...")
                cap.release()
                cap = None
                edge_client.update_active_status(self.camera_id, False)
                time.sleep(1)
                continue

            with self._frame_lock:
                self._latest_frame = frame

        if cap: cap.release()

    def _ai_loop(self):
        """Luồng xử lý AI: Chạy theo tốc độ của GPU/CPU."""
        frame_count = 0
        while self._running:
            if not self._active or not self._attendance_enabled:
                self._last_known_faces = []
                time.sleep(0.5)
                continue
                
            frame = None
            with self._frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()
            
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            # Xử lý mỗi N frame để tiết kiệm tài nguyên
            if frame_count % edge_config.process_every_n != 0:
                time.sleep(0.01)
                continue

            try:
                # 1. Phát hiện & Nhận diện
                detected = face_engine.detect_faces(frame)
                if detected:
                    cache = edge_client.get_cache(self.camera_id)
                    results = face_engine.recognize_batch(detected, cache)
                    
                    new_known_faces = []
                    for i, res in enumerate(results):
                        is_real = True
                        spoof_score = 1.0
                        
                        # Anti-Spoofing CHỈ chạy khi điểm danh đang bật
                        # Nếu chỉ xem Preview (chưa START), bỏ qua để tránh spam log
                        if self._attendance_enabled and ANTI_SPOOF_AVAILABLE and anti_spoof_service:
                            is_real, spoof_score = anti_spoof_service.is_real(frame, res.bbox)
                            res.is_real = is_real
                            res.spoof_score = spoof_score
                        
                        color_val = "unknown"
                        if res.recognized:
                            color_val = "success" if res.is_real else "danger"
                        
                        new_known_faces.append({
                            "bbox": detected[i].bbox,
                            "name": res.display_name if res.recognized else "Unknown",
                            "color_type": color_val
                        })

                        # Xử lý điểm danh (chỉ khi là người thật VÀ đã nhận diện được)
                        if self._attendance_enabled and res.is_real and res.recognized:
                            current_count = self._real_face_history.get(res.student_id, 0)
                            self._real_face_history[res.student_id] = current_count + 1
                            
                            if self._real_face_history[res.student_id] >= 3:
                                remaining = edge_client.check_cooldown(res.student_id, self.camera_id)
                                if remaining <= 0:
                                    edge_client.send_attendance(
                                        embedding=detected[i].embedding,
                                        camera_id=self.camera_id,
                                        liveness_score=spoof_score,
                                        liveness_checked=ANTI_SPOOF_AVAILABLE,
                                    )
                                    edge_client.set_cooldown(res.student_id, self.camera_id)
                                    self._real_face_history[res.student_id] = 0
                        elif self._attendance_enabled and not res.is_real and res.recognized:
                            # Chỉ log spoof cho người đã nhận diện được (không log Unknown)
                            self._log_spoof(res)
                            self._real_face_history[res.student_id] = 0

                    self._last_known_faces = new_known_faces
                else:
                    self._last_known_faces = []
            except Exception as e:
                logger.error(f"❌ AI Loop Error [{self.camera_id}]: {e}")
            
            time.sleep(0.005)

    def _live_loop(self):
        """Luồng Live View: Vẽ và gửi ảnh lên Server (Tách riêng khỏi AI)."""
        last_upload = 0
        while self._running:
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
                headers = {"X-API-Key": edge_client._headers()["X-API-Key"]}
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
            logger.warning("⚠️ Không tìm thấy 'camera_list' trong cấu hình, dùng chế độ Single Cam fallback.")
            cam_list = [{"id": edge_config.camera_id, "name": "Default Cam", "source": edge_config.camera_source}]

        # TẮT TỰ ĐỘNG CHẠY: Ép cứng trạng thái ban đầu là False (Chờ lệnh)
        is_auto = False 
        logger.info("⏸️ Mini PC đã sẵn sàng và đang CHỜ LỆNH. Hãy bấm nút Bắt đầu trên Server...")
        
        for cam in cam_list:
            cid = cam["id"]
            src = cam["source"]
            worker = CameraWorker(cid, src)
            worker.set_active(is_auto)
            worker.set_attendance_enabled(is_auto)
            self._workers[cid] = worker
            worker.start()
            time.sleep(0.2)

        self._running = True
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
                                new_worker = CameraWorker(camera_id=target_key, source=old_worker.source)
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
                            else:
                                logger.info("🔴 NHẬN LỆNH [STOP]: Tạm dừng điểm danh, giải phóng Camera.")
                                face_engine.unload_model()

                            # Đẩy lệnh xuống điều khiển tất cả các luồng camera
                            for worker in self._workers.values():
                                worker.set_active(is_start)
                                worker.set_attendance_enabled(is_start)

                    # Cập nhật xem Server có đang muốn xem trước (Preview) camera nào không
                    self._target_camera_view = actual_target
                    
                    # [NEW] Khởi tạo Worker cho TẤT CẢ các camera thuộc Server
                    all_cams = cmd_data.get("all_cameras", [])
                    
                    # Đảm bảo target_camera cũng được khởi tạo (fallback nếu server ko gửi all_cameras)
                    if actual_target and isinstance(actual_target, str):
                        if actual_target not in all_cams:
                            all_cams.append(actual_target)
                            
                    for cam_url in all_cams:
                        if isinstance(cam_url, str):
                            source_exists = any(w.source == cam_url for w in self._workers.values())
                            if cam_url not in self._workers and not source_exists:
                                logger.info(f"✨ Khởi tạo on-the-fly Worker cho Camera: {cam_url}")
                                new_worker = CameraWorker(camera_id=cam_url, source=cam_url)
                                is_sys_start = (self._current_command == "START")
                                new_worker.set_active(is_sys_start) 
                                new_worker.set_attendance_enabled(is_sys_start) 
                                self._workers[cam_url] = new_worker
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

    def stop(self):
        self._running = False
        for worker in self._workers.values():
            worker._running = False

headless_processor = HeadlessProcessor()