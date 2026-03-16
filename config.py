"""
config.py — Cấu hình trung tâm toàn bộ hệ thống
Tất cả các thông số quan trọng đều nằm ở đây.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env file nếu có
load_dotenv()

# ─────────────────────────────────────────────
#  ĐƯỜNG DẪN CƠ BẢN
# ─────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
MODELS_DIR    = BASE_DIR / "models"
ASSETS_DIR    = BASE_DIR / "assets"
SNAPSHOTS_DIR = ASSETS_DIR / "snapshots"
LOGS_DIR      = BASE_DIR / "logs"
REPORTS_DIR   = BASE_DIR / "reports" / "output"

# Tạo thư mục nếu chưa có
for _dir in [MODELS_DIR, SNAPSHOTS_DIR, LOGS_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
#  DATABASE — SQL SERVER
# ─────────────────────────────────────────────
@dataclass
class DatabaseConfig:
    # Thay đổi SERVER_NAME theo máy của bạn
    # Ví dụ: "DESKTOP-ABC123\\SQLEXPRESS" hoặc "localhost\\SQLEXPRESS"
    server:   str = os.getenv("DB_SERVER",   r"localhost\SQLEXPRESS")
    database: str = os.getenv("DB_NAME",     "FaceAttendanceDB")
    driver:   str = os.getenv("DB_DRIVER",   "ODBC Driver 17 for SQL Server")
    
    # Windows Authentication (khuyến nghị — không cần mật khẩu)
    use_windows_auth: bool = True
    
    # SQL Server Authentication (nếu không dùng Windows Auth)
    username: str = os.getenv("DB_USER", "sa")
    password: str = os.getenv("DB_PASS", "")

    @property
    def connection_string(self) -> str:
        if self.use_windows_auth:
            return (
                f"DRIVER={{{self.driver}}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"Trusted_Connection=yes;"
                f"TrustServerCertificate=yes;"
            )
        return (
            f"DRIVER={{{self.driver}}};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"UID={self.username};"
            f"PWD={self.password};"
            f"TrustServerCertificate=yes;"
        )


# ─────────────────────────────────────────────
#  AI / NHẬN DẠNG KHUÔN MẶT
# ─────────────────────────────────────────────
@dataclass
class AIConfig:
    # Model InsightFace (buffalo_l = tốt nhất, buffalo_sc = nhẹ hơn)
    model_name:       str   = "buffalo_l"
    model_pack_dir:   Path  = MODELS_DIR

    # GPU context:
    #   0  = GPU đầu tiên (yêu cầu CUDA Compute >= 6.0)
    #   -1 = CPU (dùng cho 940MX vì chỉ Compute 5.0)
    # NVIDIA 940MX → Compute 5.0 → KHÔNG đủ cho một số CUDA kernel
    # → Dùng CPU vẫn đạt 76ms (đủ nhanh cho yêu cầu ≤1s)
    gpu_ctx_id:       int   = -1   # ← FORCE CPU cho 940MX

    # ONNX Execution providers theo thứ tự ưu tiên
    # Dùng CPU only để tránh cudaErrorNoKernelImageForDevice
    onnx_providers: list = field(
        default_factory=lambda: ["CPUExecutionProvider"]
    )

    # Kích thước ảnh đầu vào cho detection
    det_size:         tuple = (640, 640)

    # Ngưỡng nhận diện (0.0 - 1.0)
    # >= 0.65 = match, < 0.65 = unknown
    recognition_threshold: float = 0.65

    # Số ảnh tối thiểu khi enroll (đăng ký) mỗi học viên
    min_enroll_photos:  int = 5
    max_enroll_photos:  int = 10

    # Kích thước embedding vector (ArcFace = 512)
    embedding_size:   int   = 512

    # Thời gian cooldown chống điểm danh trùng (giây)
    attendance_cooldown_sec: int = 60

    # Độ tin cậy tối thiểu để chụp ảnh (det_score)
    min_face_det_score: float = 0.85


# ─────────────────────────────────────────────
#  CAMERA
# ─────────────────────────────────────────────
@dataclass
class CameraConfig:
    # Mặc định: webcam USB (index 0)
    # Thay bằng RTSP URL cho camera IP qua LAN:
    #   "rtsp://admin:password@192.168.1.100:554/stream"
    source: str = "0"

    # FPS khi capture
    fps: int = 25

    # Độ phân giải
    width:  int = 1280
    height: int = 720

    # Timeout kết nối lại (giây)
    reconnect_delay_sec: int = 3
    max_reconnect_tries: int = 5

    # Số frame bỏ qua giữa các lần nhận diện
    # (1 = xử lý mọi frame, 3 = xử lý 1 trong 3 frames)
    process_every_n_frames: int = 2

    @property
    def is_ip_camera(self) -> bool:
        return str(self.source).startswith("rtsp://") or \
               str(self.source).startswith("http://")


# ─────────────────────────────────────────────
#  MULTI-CAMERA (nhiều camera theo tầng)
# ─────────────────────────────────────────────
CAMERAS: list[dict] = [
    {
        "id": 1,
        "name": "Camera Tầng 1 - Sảnh",
        "source": "rtsp://admin:admin123@192.168.1.101:554/Streaming/Channels/101",
        "floor": 1,
        "active": True,
    },
    {
        "id": 2,
        "name": "Camera Tầng 2 - Phòng học A",
        "source": "rtsp://admin:admin123@192.168.1.102:554/Streaming/Channels/101",
        "floor": 2,
        "active": True,
    },
    # Thêm camera tại đây...
    # {"id": 3, "name": "Camera Tầng 3", "source": "rtsp://...", "floor": 3, "active": True}
]


# ─────────────────────────────────────────────
#  BÁO CÁO (REPORTS)
# ─────────────────────────────────────────────
@dataclass
class ReportConfig:
    output_dir:        Path  = REPORTS_DIR
    institution_name:  str   = "Trung tâm Đào tạo XYZ"       # Thay tên cơ sở
    institution_logo:  str   = ""                              # Đường dẫn logo (tuỳ chọn)
    excel_template:    str   = "default"
    date_format:       str   = "%d/%m/%Y"
    datetime_format:   str   = "%d/%m/%Y %H:%M:%S"


# ─────────────────────────────────────────────
#  ỨNG DỤNG
# ─────────────────────────────────────────────
@dataclass
class AppConfig:
    app_name:    str  = "Hệ thống Điểm danh Nhận dạng Khuôn mặt"
    app_version: str  = "1.0.0"
    debug_mode:  bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level:   str  = "DEBUG" if debug_mode else "INFO"
    log_dir:     Path = LOGS_DIR
    
    # Snapshot: lưu ảnh khuôn mặt khi điểm danh
    save_snapshots: bool = True
    snapshot_dir:   Path = SNAPSHOTS_DIR

    # Kích thước cửa sổ chính
    window_width:  int = 1280
    window_height: int = 800


# ─────────────────────────────────────────────
#  INSTANCES (dùng trong toàn bộ project)
# ─────────────────────────────────────────────
db_config     = DatabaseConfig()
ai_config     = AIConfig()
camera_config = CameraConfig()
report_config = ReportConfig()
app_config    = AppConfig()