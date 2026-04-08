import os
import requests
from pathlib import Path
from loguru import logger

# --- Cấu hình đường dẫn ---
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "Silent-Face-Anti-Spoofing-master" / "resources" / "anti_spoof_models"

# Tự động tạo thư mục nếu chưa có
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# URL gốc từ GitHub chính thức
BASE_URL = "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models"

# Danh sách các tệp mô hình cần thiết (Đã được cập nhật tên chính xác)
MODELS = [
    "2.7_80x80_MiniFASNetV2.pth",
    "4_0_0_80x80_MiniFASNetV1SE.pth",
]

def download_models():
    """
    Tải bộ lọc Anti-Spoofing (3 models) về Mini PC.
    Sử dụng requests để tải và ghi file nhị phân.
    """
    logger.info("🚀 Đang khởi tạo quá trình tải bộ lọc Anti-Spoofing...")
    logger.info(f"📁 Thư mục đích: {MODELS_DIR}")
    
    success_count = 0
    
    for model_name in MODELS:
        dest_path = MODELS_DIR / model_name
        
        # Nếu đã có file thì bỏ qua và kiểm tra kích thước tối thiểu
        if dest_path.exists() and dest_path.stat().st_size > 1000 * 1024: # Ít nhất 1MB cho .pth
            logger.info(f"✅ Đã có sẵn: {model_name} (Bỏ qua)")
            success_count += 1
            continue
            
        url = f"{BASE_URL}/{model_name}"
        logger.info(f"⏳ Đang tải: {model_name}...")
        
        try:
            # Tải dữ liệu theo từng chunk để không tốn RAM
            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status() # Kiểm tra lỗi HTTP (404, 500...)
            
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.success(f"✔️ Hoàn thành: {model_name}")
            success_count += 1
        except Exception as e:
            logger.error(f"❌ Lỗi khi tải {model_name}: {e}")
            if dest_path.exists():
                os.remove(dest_path) # Xoá file lỗi để tránh load lỗi sau này

    # --- Kiểm tra InsightFace Models ---
    INSIGHTFACE_DIR = BASE_DIR / "models" / "buffalo_s"
    if not INSIGHTFACE_DIR.exists() or len(list(INSIGHTFACE_DIR.glob("*.onnx"))) < 5:
        logger.warning("⚠️ Thiếu bộ model InsightFace (buffalo_s).")
        logger.info("Hãy đảm bảo bạn đã giải nén 'buffalo_s.zip' vào thư mục 'MINI_PC/models/buffalo_s/'")
    else:
        logger.success("✅ Bộ model InsightFace (buffalo_s) đã sẵn sàng.")

    if success_count == len(MODELS):
        logger.success("✨ TẤT CẢ MODEL ĐÃ SẴN SÀNG! ✨")
        print("\n--- HOÀN TẤT ---")
        print("Bạn hãy chạy lại: python main_edge.py")
    else:
        logger.warning(f"⚠️ Chỉ tải được {success_count}/{len(MODELS)} tệp. Vui lòng kiểm tra kết nối mạng và thử lại.")

if __name__ == "__main__":
    download_models()
