"""
main.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Entry point — chạy ứng dụng FaceAttend

Cách khởi động:
    cd C:\face_attendance
    py -3.11 main.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys
from pathlib import Path

# Đảm bảo thư mục gốc trong PATH
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Tạo các thư mục cần thiết nếu chưa có
for folder in ["logs", "models", "assets/snapshots", "assets/profiles", "reports"]:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)

from ui.main_window import run

if __name__ == "__main__":
    run()
