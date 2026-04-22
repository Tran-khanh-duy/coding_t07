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
from services.face_engine import face_engine, ANTI_SPOOF_AVAILABLE
try:
    from services.anti_spoof_service import anti_spoof_service
except Exception as e:
    logger.warning(f"AntiSpoofPredict không khả dụng: {e}")
    anti_spoof_service = None
    ANTI_SPOOF_AVAILABLE = False

from edge_client import edge_client

# Registry toàn cục để theo dõi các cổng phần cứng đang bận
os.environ["OPENCV_VIDEOIO_PRIORITY_OBSENSOR"] = "0" 
ACTIVE_SOURCES = set()
SOURCES_LOCK = threading.Lock()

class CameraWorker(threading.Thread):
    """Luồng xử lý cho một camera cụ thể."""
    
    def __init__(self, camera_id: str, source: str):
        super().__init__(name=f"Worker-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.source = source
        self._running = False
        self._cap = None
        self._active = False 
        self._is_previewing = False 
        self._attendance_enabled = False 
        
        # Trạng thái truyền tải
        self._is_uploading = False
        self._spoof_log_cache = {}
        
        # "Trí nhớ ngắn hạn" để giữ tên hiển thị mượt mà trên UI khi bỏ qua frame
        self._last_known_faces = []

    def set_active(self, active: bool):
        self._active = active

    def set_previewing(self, previewing: bool):
        self._is_previewing = previewing

    def set_attendance_enabled(self, enabled: bool):
        # Chỉ những camera là RTSP mới được phép bật nhận diện (attendance)
        if enabled and str(self.source).startswith("rtsp://"):
            self._attendance_enabled = True
        else:
            self._attendance_enabled = False

    def run(self):
        self._running = True
        frame_count = 0
        last_frame_upload = 0.0
        
        logger.info(f"🚀 Bắt đầu Worker thread cho camera: {self.camera_id} (Source: {self.source})")
        
        while self._running:
            if not self._active and not self._is_previewing:
                if self._cap:
                    self._cap.release()
                    self._cap = None
                    edge_client.update_active_status(self.camera_id, False)
                    with SOURCES_LOCK:
                        if self.source in ACTIVE_SOURCES:
                            ACTIVE_SOURCES.remove(self.source)
                    logger.info(f"⏹️ Camera {self.camera_id}: Đã giải phóng tài nguyên {self.source}")
                time.sleep(1)
                continue

            if self._cap is None or not self._cap.isOpened():
                self._open_camera()
                if self._cap is None or not self._cap.isOpened():
                    edge_client.update_active_status(self.camera_id, False)
                    time.sleep(2)
                    continue
            
            edge_client.update_active_status(self.camera_id, True)

            ret, frame = self._cap.read()
            if not ret:
                logger.warning(f"⚠️ Camera {self.camera_id}: Mất tín hiệu, đang thử lại...")
                edge_client.update_active_status(self.camera_id, False)
                if self._cap:
                    self._cap.release()
                    self._cap = None
                with SOURCES_LOCK:
                    if hasattr(self, 'source') and self.source in ACTIVE_SOURCES:
                        ACTIVE_SOURCES.remove(self.source)
                time.sleep(1)
                continue

            frame_count += 1
            now = time.time()

            is_recognition_step = (frame_count % edge_config.process_every_n == 0) and self._active and self._attendance_enabled
            is_preview_step = self._is_previewing and (now - last_frame_upload >= 0.1)

            if is_recognition_step or is_preview_step:
                try:
                    # 1. Phát hiện khuôn mặt
                    detected = face_engine.detect_faces(frame)
                    
                    # 2. Nhận diện (Chỉ thực hiện ở recognition step)
                    if is_recognition_step and detected:
                        cache = edge_client.get_cache()
                        results = face_engine.recognize_batch(detected, cache)
                        
                        # Xóa trí nhớ cũ trước khi cập nhật kết quả nhận diện mới nhất
                        self._last_known_faces = []
                        
                        for i, res in enumerate(results):
                            is_real = True
                            spoof_score = 1.0
                            if ANTI_SPOOF_AVAILABLE and anti_spoof_service:
                                is_real, spoof_score = anti_spoof_service.is_real(frame, res.bbox)
                                res.is_real = is_real
                                res.spoof_score = spoof_score
                            
                            # Ghi nhớ khuôn mặt này để vẽ khung trên UI
                            if res.recognized:
                                color_val = "success" if res.is_real else "danger"
                                self._last_known_faces.append({
                                    "bbox": detected[i].bbox,
                                    "name": res.full_name,
                                    "color_type": color_val
                                })
                            else:
                                self._last_known_faces.append({
                                    "bbox": detected[i].bbox,
                                    "name": "Unknown",
                                    "color_type": "unknown"
                                })

                            if not res.is_real:
                                self._log_spoof(res)
                                continue

                            if res.recognized:
                                remaining = edge_client.check_cooldown(res.student_id)
                                if remaining <= 0:
                                    logger.info(f"👤 [{self.camera_id}] Nhận diện: {res.full_name}")
                                    edge_client.send_attendance(
                                        embedding=detected[i].embedding,
                                        liveness_score=spoof_score,
                                        liveness_checked=ANTI_SPOOF_AVAILABLE,
                                    )
                                    edge_client.set_cooldown(res.student_id)

                    # 3. Gửi Frame cho Server (Nếu đang Preview)
                    if is_preview_step:
                        last_frame_upload = now
                        dets_payload = []
                        if detected:
                            for det in detected:
                                name = "Unknown"
                                color_type = "unknown"
                                
                                # Tracking nội suy: Tìm khuôn mặt trong "trí nhớ" khớp với vị trí hiện tại
                                cx = (det.bbox[0] + det.bbox[2]) / 2
                                cy = (det.bbox[1] + det.bbox[3]) / 2
                                min_dist = float('inf')
                                best_match = None
                                
                                for known in self._last_known_faces:
                                    kx = (known['bbox'][0] + known['bbox'][2]) / 2
                                    ky = (known['bbox'][1] + known['bbox'][3]) / 2
                                    dist = ((cx - kx)**2 + (cy - ky)**2)**0.5
                                    if dist < 150 and dist < min_dist:
                                        min_dist = dist
                                        best_match = known
                                
                                if best_match:
                                    name = best_match['name']
                                    color_type = best_match['color_type']

                                dets_payload.append([
                                    int(det.bbox[0]), int(det.bbox[1]), 
                                    int(det.bbox[2]), int(det.bbox[3]),
                                    name, color_type
                                ])
                        self._upload_frame_async(frame, dets_payload)

                except Exception as e:
                    logger.error(f"❌ [{self.camera_id}] Lỗi AI Pipeline: {e}")

        if self._cap:
            self._cap.release()

    def _open_camera(self):
        source = self.source
        try:
            initial_src = int(source)
        except (ValueError, TypeError):
            initial_src = source
            
        search_queue = [initial_src]
        if isinstance(initial_src, int):
            for alt in range(11):
                if alt not in search_queue:
                    search_queue.append(alt)
        
        for cam_source in search_queue:
            with SOURCES_LOCK:
                if cam_source in ACTIVE_SOURCES:
                    continue
                ACTIVE_SOURCES.add(cam_source)
            
            logger.info(f"📸 [{self.camera_id}] Thử mở Camera: source='{cam_source}'")
            try:
                if isinstance(cam_source, int):
                    self._cap = cv2.VideoCapture(cam_source, cv2.CAP_DSHOW)
                    if not (self._cap and self._cap.isOpened()):
                        self._cap = cv2.VideoCapture(cam_source, cv2.CAP_ANY)
                else:
                    self._cap = cv2.VideoCapture(cam_source, cv2.CAP_FFMPEG)
                    if not (self._cap and self._cap.isOpened()):
                        self._cap = cv2.VideoCapture(cam_source, cv2.CAP_ANY)

                if self._cap and self._cap.isOpened():
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.width)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.height)
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.source = cam_source 
                    logger.info(f"📸 [{self.camera_id}] Đã mở thành công tại source {cam_source}.")
                    return
                else:
                    with SOURCES_LOCK:
                        ACTIVE_SOURCES.remove(cam_source)
            except Exception as e:
                logger.error(f"❌ [{self.camera_id}] Lỗi khi mở {cam_source}: {e}")
                with SOURCES_LOCK:
                    if cam_source in ACTIVE_SOURCES:
                        ACTIVE_SOURCES.remove(cam_source)
            
        logger.error(f"❌ [{self.camera_id}] Thất bại hoàn toàn sau khi thử các cổng {search_queue}")

    def _upload_frame_async(self, frame, dets=None):
        if self._is_uploading:
            return 
            
        cam_id = self.camera_id
        def task():
            self._is_uploading = True
            try:
                # Copy frame để tránh xung đột luồng
                frame_copy = frame.copy()
                
                if dets:
                    # 1. Vẽ viền Bounding Box bằng OpenCV (tốc độ cao)
                    for det in dets:
                        x1, y1, x2, y2, name, color_type = det
                        if color_type == "success": color_cv = (0, 255, 0)   
                        elif color_type == "danger": color_cv = (0, 0, 255)   
                        else: color_cv = (0, 255, 255) 
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color_cv, 4)

                    # 2. Vẽ Chữ Tiếng Việt bằng Pillow (PIL)
                    frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    draw = ImageDraw.Draw(pil_img)

                    # TĂNG CỠ CHỮ LÊN 1.5 LẦN VÀ DÙNG FONT ĐẬM
                    font_size = 52
                    try:
                        font = ImageFont.truetype("arialbd.ttf", font_size)
                    except IOError:
                        try:
                            font = ImageFont.truetype("arial.ttf", font_size)
                        except IOError:
                            font = ImageFont.load_default()

                    for det in dets:
                        x1, y1, x2, y2, name, color_type = det
                        if color_type == "success": color_pil = (0, 255, 0)   
                        elif color_type == "danger": color_pil = (255, 0, 0)   
                        else: color_pil = (255, 255, 0)

                        if hasattr(font, 'getbbox'):
                            bbox = font.getbbox(name)
                            txt_w = bbox[2] - bbox[0]
                            txt_h = bbox[3] - bbox[1]
                        else:
                            txt_w, txt_h = draw.textsize(name, font=font)
                        
                        # Tính bù trừ ngược đảm bảo tọa độ không bao giờ bị âm hoặc y1 < y0
                        bg_x1 = max(0, int(x1))
                        bg_y1 = max(0, int(y1) - int(txt_h) - 20)
                        bg_x2 = int(bg_x1 + txt_w + 20)
                        bg_y2 = int(bg_y1 + txt_h + 20)
                        
                        draw.rectangle([(bg_x1, bg_y1), (bg_x2, bg_y2)], fill=(0, 0, 0))
                        draw.text((bg_x1 + 10, bg_y1 + 4), name, font=font, fill=color_pil)

                    frame_copy = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                # Thu nhỏ ảnh để tiết kiệm băng thông
                h, w = frame_copy.shape[:2]
                sw = 640
                sh = int(h * (sw / w))
                small = cv2.resize(frame_copy, (sw, sh))
                _, buffer = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
                img_b64 = base64.b64encode(buffer).decode("utf-8")
                
                url = f"{edge_client.server_url}/api/system/frame"
                payload = {
                    "image_b64": img_b64, 
                    "camera_id": cam_id,
                    "detections": dets or []
                }
                headers = {"X-API-Key": edge_client._headers()["X-API-Key"]}
                
                resp = requests.post(url, json=payload, headers=headers, timeout=10)
                if resp.status_code != 200:
                    logger.debug(f"📤 [{cam_id}] Upload frame fail: {resp.status_code}")
                # else:
                #     logger.debug(f"📤 [{cam_id}] Upload frame SUCCESS!")
            except Exception as e:
                logger.error(f"📤 [{cam_id}] Lỗi upload frame: {e}")
            finally:
                self._is_uploading = False
        
        threading.Thread(target=task, daemon=True).start()

    def _log_spoof(self, res):
        now = time.time()
        key = res.student_id if res.recognized else "unknown"
        if now - self._spoof_log_cache.get(key, 0) > 5.0:
            logger.warning(f"🚫 [{self.camera_id}] Phát hiện GIẢ MẠO: {res.display_name}")
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
            
            # Cứ mỗi 2 giây, Mini PC sẽ gọi API lên Server để "hỏi" xem có lệnh mới không
            if now - last_command_check >= 2.0:
                last_command_check = now
                try:
                    cmd_data = edge_client.get_system_command_raw()
                    new_cmd = cmd_data.get("command", "STOP")
                    actual_target = cmd_data.get("target_camera")

                    # NẾU PHÁT HIỆN LỆNH MỚI TỪ SERVER
                    if new_cmd != self._current_command:
                        self._current_command = new_cmd
                        is_start = (new_cmd == "START")
                        
                        # In log thông báo trạng thái
                        if is_start:
                            logger.info("🟢 NHẬN LỆNH [START]: Đánh thức Camera, bắt đầu điểm danh!")
                        else:
                            logger.info("🔴 NHẬN LỆNH [STOP]: Tạm dừng điểm danh, giải phóng Camera.")

                        # Đẩy lệnh xuống điều khiển tất cả các luồng camera
                        for worker in self._workers.values():
                            worker.set_active(is_start)
                            worker.set_attendance_enabled(is_start)

                    # Luôn cập nhật xem Server có đang muốn xem trước (Preview) camera nào không
                    self._target_camera_view = actual_target
                    
                    if actual_target and isinstance(actual_target, str) and actual_target.startswith("rtsp://"):
                        source_exists = any(w.source == actual_target for w in self._workers.values())
                        if actual_target not in self._workers and not source_exists:
                            logger.info(f"✨ Khởi tạo on-the-fly Worker cho: {actual_target}")
                            new_worker = CameraWorker(camera_id=actual_target, source=actual_target)
                            is_sys_start = (self._current_command == "START")
                            new_worker.set_active(True) # Luôn để True để đọc ảnh preview
                            new_worker.set_attendance_enabled(is_sys_start) 
                            self._workers[actual_target] = new_worker
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

            time.sleep(0.5)

    def stop(self):
        self._running = False
        for worker in self._workers.values():
            worker._running = False

headless_processor = HeadlessProcessor()