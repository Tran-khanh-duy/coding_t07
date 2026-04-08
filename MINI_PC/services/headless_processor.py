"""
services/headless_processor.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Headless AI Processor for Mini PC (Edge Box)
- Core loop: Capture -> Detect -> Anti-Spoof -> Recognize -> Send
- No UI dependencies, uses loguru for feedback.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import time
import cv2
import numpy as np
import threading
from loguru import logger
from pathlib import Path

from config import edge_config, ai_config, anti_spoof_config
from services.face_engine import face_engine, ANTI_SPOOF_AVAILABLE
try:
    from services.anti_spoof_service import anti_spoof_service
except Exception as e:
    logger.warning(f"AntiSpoofPredict không khả dụng: {e}")
    anti_spoof_service = None
    ANTI_SPOOF_AVAILABLE = False

from edge_client import edge_client

class HeadlessProcessor:
    """Standalone AI processor without UI signals."""
    
    def __init__(self):
        self._running = False
        self._cap = None
        self._latest_ai_results = []
        self._ai_lock = threading.Lock()
        
        # Cache để giảm rác log (chống spam log spoofing)
        self._spoof_log_cache = {} # {id/name: last_log_time}
        
    def start(self):
        """Khởi động vòng lặp xử lý AI."""
        if self._running:
            return
            
        logger.info("🚀 Headless Processor đang khởi động...")
        
        # 1. Load Model
        ai_config.onnx_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if not face_engine.load_model():
            logger.error("❌ Không thể nạp model AI. Dừng xử lý.")
            return

        # 2. Đồng bộ ban đầu
        logger.info("📥 Đang kéo danh sách học viên từ Server...")
        if edge_client.pull_embeddings():
            cache = edge_client.get_cache()
            logger.success(f"✅ Đã tải {cache.size} học viên")
        else:
            logger.warning("⚠️ Không thể tải dữ liệu từ Server, sẽ sử dụng cache cũ hoặc chờ đồng bộ.")

        self._running = True
        self._run_loop()

    def _run_loop(self):
        """Vòng lặp chính với máy trạng thái (Command-based)."""
        frame_count = 0
        last_embed_check = time.time()
        last_command_check = 0
        self._current_command = "START" if edge_config.auto_start else "STOP"
        self._last_frame_upload = 0.0

        if edge_config.auto_start:
            logger.info("▶️ TỰ ĐỘNG KÍCH HOẠT: BẮT ĐẦU ĐIỂM DANH")
        else:
            logger.info("📡 Đang chuyển vào chế độ chờ lệnh từ Server...")

        while self._running:
            now = time.time()

            # 1. Kiểm tra lệnh từ Server mỗi 2 giây
            if now - last_command_check >= 2.0:
                last_command_check = now
                new_cmd = edge_client.get_system_command()
                
                if new_cmd != self._current_command:
                    self._current_command = new_cmd
                    if self._current_command == "START":
                        logger.success("▶️ NHẬN LỆNH: BẮT ĐẦU ĐIỂM DANH")
                        # Thử mở camera (có retry nếu lỗi)
                        source = edge_config.camera_source
                        cam_source = int(source) if source.isdigit() else source
                        
                        retry_count = 0
                        while self._cap is None or not self._cap.isOpened():
                            # Optimization: Sử dụng DirectShow trên Windows cho Webcams (0, 1...) giúp mở camera nhanh hơn
                            backend = cv2.CAP_DSHOW if isinstance(cam_source, int) else cv2.CAP_ANY
                            
                            self._cap = cv2.VideoCapture(cam_source, backend)
                            if self._cap.isOpened():
                                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                                # Giảm buffer xuống 1 để hình ảnh luôn mới nhất
                                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
                                logger.info(f"📸 Đã mở Camera: {source} (Backend: {backend})")
                                break
                            else:
                                retry_count += 1
                                logger.warning(f"⚠️ Không thể mở camera {source} (Lần {retry_count}). Thử lại sau 2 giây...")
                                time.sleep(2)
                                if not self._running or self._current_command == "STOP":
                                    break
                    else:
                        logger.warning("⏹️ NHẬN LỆNH: KẾT THÚC ĐIỂM DANH")
                        # Giải phóng camera để tiết kiệm tài nguyên
                        if self._cap and self._cap.isOpened():
                            self._cap.release()
                            self._cap = None
                            logger.info("🔌 Đã giải phóng Camera. Chế độ chờ...")

            # 2. Nếu đang ở trạng thái STOP, chỉ nghỉ rồi tiếp tục poll
            if self._current_command == "STOP":
                time.sleep(0.5)
                continue

            # 3. Nếu đang ở trạng thái START, tiến hành xử lý frame
            if self._cap is None or not self._cap.isOpened():
                time.sleep(1)
                continue

            ret, frame = self._cap.read()
            if not ret:
                logger.warning("⚠️ Mất tín hiệu camera, đang thử lại...")
                time.sleep(1)
                continue

            frame_count += 1
            
            # Refresh embeddings định kỳ (mỗi 1 phút)
            if now - last_embed_check >= 60:
                last_embed_check = now
                if edge_client.should_refresh_embeddings():
                    logger.info("🔄 Đang cập nhật danh sách học viên mới...")
                    edge_client.pull_embeddings()

            # Chỉ xử lý mỗi N frame
            if frame_count % edge_config.process_every_n != 0:
                pass # Vẫn tiếp tục để kiểm tra upload frame bên dưới
            
            # ── Upload frame cho Server UI xem (Remote View) ──
            # Target khoảng 5-10 FPS
            # Target khoảng 20 FPS
            if now - self._last_frame_upload >= 0.05: # ~20 FPS (Tăng tốc mượt hơn)
                self._last_frame_upload = now
                self._upload_frame_async(frame)

            if frame_count % edge_config.process_every_n != 0:
                continue

            # ── AI Pipeline ──
            cache = edge_client.get_cache()
            detected = face_engine.detect_faces(frame)

            if not detected:
                continue

            # Nhận diện hàng loạt (arcface)
            results = face_engine.recognize_batch(detected, cache)
            
            # --- KIỂM TRA ANTI-SPOOFING TRƯỚC KHI CẬP NHẬT RA UI ---
            for i, res in enumerate(results):
                is_real = True
                spoof_score = 1.0
                if ANTI_SPOOF_AVAILABLE and anti_spoof_service:
                    try:
                        is_real, spoof_score = anti_spoof_service.is_real(frame, res.bbox)
                        res.is_real = is_real
                        res.spoof_score = spoof_score
                    except Exception:
                        is_real = True
            
            # Sau khi đã có đầy đủ nhận diện + spoof check mới cập nhật ra UI
            with self._ai_lock:
                self._latest_ai_results = results

            # Quay lại logic gửi attendance cho những mặt là THẬT
            for i, res in enumerate(results):
                if not res.is_real:
                    # Giới hạn tần suất in log cảnh báo (không spam)
                    now_ts = time.time()
                    name_key = res.student_id if res.recognized else f"unknown_{i}"
                    last_log = self._spoof_log_cache.get(name_key, 0)
                    
                    if now_ts - last_log >= 3.0:
                        logger.warning(f"🚫 Phát hiện GIẢ MẠO ({res.display_name}) - Score: {res.spoof_score:.3f}")
                        self._spoof_log_cache[name_key] = now_ts
                    continue

                # 2. Nhận diện
                if not res.recognized:
                    continue

                # 3. Cooldown
                remaining = edge_client.check_cooldown(res.student_id)
                if remaining > 0:
                    continue

                # 4. Gửi kết quả
                logger.info(f"👤 Nhận diện: {res.full_name} ({res.similarity*100:.1f}%)")
                
                embedding_to_send = detected[i].embedding
                if embedding_to_send is not None:
                    response = edge_client.send_attendance(
                        embedding=embedding_to_send,
                        liveness_score=spoof_score,
                        liveness_checked=ANTI_SPOOF_AVAILABLE,
                    )
                    
                    edge_client.set_cooldown(res.student_id)
                    
                    status = response.get("status", "")
                    if status == "success":
                        logger.success(f"✅ Điểm danh THÀNH CÔNG: {res.full_name}")
                    elif status == "offline":
                        logger.info(f"💾 Đã lưu dữ liệu offline cho: {res.full_name}")

        if self._cap and self._cap.isOpened():
            self._cap.release()
        logger.info("Headless Processor đã dừng.")

    def _upload_frame_async(self, frame):
        """Mã hoá và gửi khung hình JPEG lên server (Non-blocking)."""
        import threading
        import base64

        def task():
            try:
                # 1. Lấy kết quả AI mới nhất để vẽ
                with self._ai_lock:
                    results = self._latest_ai_results.copy()
                
                # 2. Vẽ detection boxes lên frame
                # Note: draw_results sẽ tự clone frame
                vis_frame = face_engine.draw_results(frame, results)

                # 3. Resize để tăng độ nét (Tăng lên 960px)
                h, w = vis_frame.shape[:2]
                new_w = 960
                new_h = int(h * (new_w / w))
                small_frame = cv2.resize(vis_frame, (new_w, new_h))
                
                # Tăng chất lượng JPEG từ 60 lên 85
                _, buffer = cv2.imencode(".jpg", small_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_b64 = base64.b64encode(buffer).decode("utf-8")
                
                # Gửi qua edge_client hoặc requests trực tiếp
                from config import edge_config
                url = f"{edge_config.server_url.rstrip('/')}/api/system/frame"
                headers = {
                    "X-API-Key": edge_config.api_key,
                    "Content-Type": "application/json"
                }
                requests.post(url, json={"image_b64": img_b64}, headers=headers, timeout=2)
            except Exception:
                pass # Bỏ qua lỗi upload để không crash loop AI

        import requests # Cần import trong thread hoặc bọc lại
        threading.Thread(target=task, daemon=True).start()

    def stop(self):
        """Dừng xử lý AI."""
        self._running = False

# Export instance
headless_processor = HeadlessProcessor()
