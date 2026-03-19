import cv2
import sys
from pathlib import Path

# Thêm root vào path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from services.anti_spoof_service import anti_spoof_service
from loguru import logger

def test_on_image(img_path):
    logger.info(f"Đang test ảnh: {img_path}")
    frame = cv2.imread(img_path)
    if frame is None:
        logger.error(f"Không thể load ảnh {img_path}")
        return

    # Giả lập bbox (toàn bộ ảnh hoặc trung tâm)
    h, w = frame.shape[:2]
    bbox = [10, 10, w-20, h-20]
    
    is_real, score = anti_spoof_service.is_real(frame, bbox)
    logger.info(f"Kết quả: {'THẬT' if is_real else 'GIẢ'} (Score: {score:.4f})")

if __name__ == "__main__":
    # Test với ảnh mẫu nếu có
    sample_path = root / "test_frame.jpg"
    if sample_path.exists():
        test_on_image(str(sample_path))
    else:
        logger.warning(f"Không tìm thấy ảnh mẫu {sample_path}")
    
    # Test với ảnh trong Silent-Face-Anti-Spoofing-master/images/sample nếu có
    sample_dir = root / "Silent-Face-Anti-Spoofing-master" / "images" / "sample"
    if sample_dir.exists():
        for img_file in sample_dir.glob("*.jpg"):
            test_on_image(str(img_file))
