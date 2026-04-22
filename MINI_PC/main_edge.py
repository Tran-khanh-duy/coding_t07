"""
main_edge.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Entry point cho Mini PC (Edge Box)
Chạy giao diện Kiosk — chỉ có camera feed + nhận diện

Cách khởi động:
    python main_edge.py

Cấu hình:
    Sửa file .env.edge hoặc truyền biến môi trường
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os
import sys
import time
from pathlib import Path
from loguru import logger

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Tối ưu kết nối siêu tốc cho Camera IP (bỏ qua bước phân tích stream rườm rà)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|analyzeduration;500000|probesize;5000000"
# Load .env.edge nếu tồn tại
ROOT = Path(__file__).parent
env_file = ROOT / ".env.edge"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file, override=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Server"))

# Tạo các thư mục cần thiết
for folder in ["logs", "models", "database"]:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)

from config import edge_config
from headless_processor import headless_processor
from edge_client import edge_client

def run_headless():
    """Chạy Edge AI ở chế độ Headless (Không giao diện)."""
    
    print("=" * 60)
    print(f"  FACEATTEND EDGE (HEADLESS) — {edge_config.device_name}")
    print(f"  Server: {edge_config.server_url}")
    print(f"  Camera Source: {edge_config.camera_source}")
    print(f"  AI Processing mode: ACTIVE")
    print("=" * 60)

    try:
        # Khởi động xử lý AI
        headless_processor.start()
    except KeyboardInterrupt:
        logger.info("Dừng hệ thống theo yêu cầu (KeyboardInterrupt)...")
    except Exception as e:
        logger.exception(f"Lỗi nghiêm trọng: {e}")
    finally:
        # Dọn dẹp
        headless_processor.stop()
        edge_client.stop()
        logger.info("Hệ thống đã tắt an toàn.")

if __name__ == "__main__":
    run_headless()
