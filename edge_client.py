"""
edge_client.py
Mô phỏng Hộp Biên (Edge AI Box) đặt tại phòng học.
Nhiệm vụ: Đọc Camera -> Ép khuôn mặt thành Vector -> Gửi JSON về Server.
"""
import cv2
import requests
import time
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from services.face_engine import face_engine
import config

# Cấu hình kết nối tới Máy chủ API
SERVER_API_URL = "http://192.168.10.100:8000/api/attendance"
API_KEY = "faceattend_secret_2026"
CAMERA_ID = "CAM_PHONG_A101"

def run_edge_box():
    logger.info("🛡️ EDGE BOX ĐANG KHỞI ĐỘNG...")
    
    # 1. Ép AI chạy bằng CPU để giả lập sức mạnh của Mini PC
    config.ai_config.onnx_providers = ["CPUExecutionProvider"]
    face_engine.load_model()
    
    # 2. Mở luồng Camera (Dùng Webcam số 0 trên máy bạn để test)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("❌ Không thể mở Webcam!")
        return

    frame_count = 0
    last_send_time = 0
    
    logger.info("✅ Edge Box đã sẵn sàng. Đưa khuôn mặt vào khung hình...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Tối ưu: Bỏ qua bớt khung hình (Chỉ xử lý 1/3 số frame để giảm tải CPU)
        frame_count += 1
        if frame_count % 3 != 0:
            cv2.imshow("EDGE BOX (Nhan 'q' de thoat)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue
            
        # 3. Phân tích khuôn mặt bằng InsightFace
        faces = face_engine.detect_faces(frame)
        
        for face in faces:
            # Lấy tọa độ để vẽ khung vuông trên màn hình
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # 4. Gửi dữ liệu về Server (Giới hạn 3 giây mới gửi 1 lần)
            now = time.time()
            if now - last_send_time > 3.0:
                vector_512 = face.embedding.tolist() # Chuyển Numpy Array thành Python List
                
                payload = {
                    "camera_id": CAMERA_ID,
                    "timestamp": str(now),
                    "embedding": vector_512,
                    "liveness_score": 0.99  # Giả lập điểm mặt thật (Vì đang thiếu model Anti-Spoof)
                }
                
                headers = {"X-API-Key": API_KEY}
                
                try:
                    logger.info("📤 Đang gửi Vector (512 số) về Server...")
                    response = requests.post(SERVER_API_URL, json=payload, headers=headers, timeout=2)
                    
                    if response.status_code == 200:
                        data = response.json()
                        # Xử lý kết quả Server trả về
                        if data["status"] == "success":
                            logger.success(f"✅ Hợp lệ: {data.get('full_name')} (Độ tin cậy: {data.get('similarity'):.2f})")
                        elif data["status"] == "unknown":
                            logger.warning("❌ Server báo: KHUÔN MẶT LẠ")
                        else:
                            logger.info(f"Server báo: {data.get('message')}")
                    else:
                        logger.error(f"Lỗi Server (Mã {response.status_code}): {response.text}")
                except Exception as e:
                    logger.error(f"❌ Mất kết nối mạng tới Server: {e}")
                    
                last_send_time = now

        cv2.imshow("EDGE BOX (Nhan 'q' de thoat)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_edge_box()