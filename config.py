"""
config.py — Cấu hình trung tâm 
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

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

for _dir in [MODELS_DIR, SNAPSHOTS_DIR, LOGS_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  DATABASE — SQL SERVER
# ─────────────────────────────────────────────
@dataclass
class DatabaseConfig:
    server:   str = os.getenv("DB_SERVER",   r"localhost\SQLEXPRESS")
    database: str = os.getenv("DB_NAME",     "FaceAttendanceDB")
    driver:   str = os.getenv("DB_DRIVER",   "ODBC Driver 17 for SQL Server")
    use_windows_auth: bool = True
    username: str = os.getenv("DB_USER", "sa")
    password: str = os.getenv("DB_PASS", "")

    @property
    def connection_string(self) -> str:
        if self.use_windows_auth:
            return (
                f"DRIVER={{{self.driver}}};SERVER={self.server};"
                f"DATABASE={self.database};Trusted_Connection=yes;"
                f"TrustServerCertificate=yes;"
            )
        return (
            f"DRIVER={{{self.driver}}};SERVER={self.server};"
            f"DATABASE={self.database};UID={self.username};"
            f"PWD={self.password};TrustServerCertificate=yes;"
        )

# ─────────────────────────────────────────────
#  AI / NHẬN DẠNG KHUÔN MẶT (CẤU HÌNH SERVER)
# ─────────────────────────────────────────────
@dataclass
class AIConfig:
    model_name:       str   = "buffalo_l"
    model_pack_dir:   Path  = MODELS_DIR

    # Sử dụng GPU
    gpu_ctx_id:       int   = 0   

    # Ép dùng CUDA cho Server
    onnx_providers: list = field(
        default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    # TRẢ LẠI CHUẨN ĐỘ PHÂN GIẢI CAO: Giữ (640, 640) để model quét kỹ hơn,
    # bắt được khuôn mặt ở xa và góc nghiêng tốt hơn, đảm bảo độ chính xác >98%.
    det_size:         tuple = (640, 640)

    recognition_threshold: float = 0.50 

    min_enroll_photos:  int = 5
    max_enroll_photos:  int = 10
    embedding_size:   int   = 512
    attendance_cooldown_sec: int = 60
    min_face_det_score: float = 0.60

# ─────────────────────────────────────────────
#  CAMERA (CẤU HÌNH SERVER)
# ─────────────────────────────────────────────
@dataclass
class CameraConfig:
    source: str = "0"
    fps: int = 25
    width:  int = 1280
    height: int = 720
    reconnect_delay_sec: int = 3
    max_reconnect_tries: int = 5

    # TĂNG TẦN SUẤT XỬ LÝ: Xử lý 1 nửa số khung hình (ví dụ 30fps thì AI quét 15fps).
    # Đủ dày đặc để không bỏ lót bất kỳ ai lướt ngang qua, mà không bị thừa thãi dữ liệu.
    process_every_n_frames: int = 2

    @property
    def is_ip_camera(self) -> bool:
        return str(self.source).startswith("rtsp://") or \
               str(self.source).startswith("http://")

CAMERAS: list[dict] = [
    {"id": 1, "name": "Camera Tầng 1", "source": "0", "floor": 1, "active": True},
]

@dataclass
class ReportConfig:
    output_dir:        Path  = REPORTS_DIR
    institution_name:  str   = "Trung tâm Đào tạo XYZ"
    institution_logo:  str   = ""
    excel_template:    str   = "default"
    date_format:       str   = "%d/%m/%Y"
    datetime_format:   str   = "%d/%m/%Y %H:%M:%S"

@dataclass
class AppConfig:
    app_name:    str  = "Hệ thống Điểm danh Nhận dạng Khuôn mặt"
    app_version: str  = "1.0.0"
    debug_mode:  bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level:   str  = "DEBUG" if debug_mode else "INFO"
    log_dir:     Path = LOGS_DIR
    save_snapshots: bool = True
    snapshot_dir:   Path = SNAPSHOTS_DIR
    window_width:  int = 1280
    window_height: int = 800

db_config     = DatabaseConfig()
ai_config     = AIConfig()
camera_config = CameraConfig()
report_config = ReportConfig()
app_config    = AppConfig()