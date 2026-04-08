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
    server:   str = os.getenv("DB_SERVER",   r".")
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
    model_name:       str   = "buffalo_s"  # Chuyển sang model nhỏ cho Edge
    model_pack_dir:   Path  = MODELS_DIR

    # Sử dụng GPU
    gpu_ctx_id:       int   = 0   

    # Ép dùng CUDA cho Server
    onnx_providers: list = field(
        default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    # TRẢ LẠI CHUẨN ĐỘ PHÂN GIẢI CAO: Giữ (640, 640) để model quét kỹ hơn,
    # bắt được khuôn mặt ở xa và góc nghiêng tốt hơn, đảm bảo độ chính xác >98%.
    det_size:         tuple = (320, 320)  # Giảm độ phân giải để tăng tốc độ phát hiện

    recognition_threshold: float = float(os.getenv("AI_THRESHOLD", "0.60"))
    min_enroll_photos:  int = 5
    max_enroll_photos:  int = 10
    embedding_size:   int   = 512
    attendance_cooldown_sec: int = int(os.getenv("ATTENDANCE_COOLDOWN", "60"))
    min_face_det_score: float = float(os.getenv("MIN_FACE_SCORE", "0.70"))


# ─────────────────────────────────────────────
#  CAMERA (CẤU HÌNH SERVER)
# ─────────────────────────────────────────────
@dataclass
class CameraConfig:
    source: str = "0"
    fps: int = 60
    width:  int = 1280
    height: int = 720
    reconnect_delay_sec: int = 3
    max_reconnect_tries: int = 5

    # TĂNG KHOẢNG CÁCH XỬ LÝ: Skip khung hình để giảm tải CPU/GPU
    process_every_n_frames: int = int(os.getenv("CAM_PROCESS_N", "3"))

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
    institution_name:  str   = os.getenv("INSTITUTION_NAME", "Trung tâm Đào tạo XYZ")
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

@dataclass
class AntiSpoofConfig:
    model_dir: Path = BASE_DIR / "Silent-Face-Anti-Spoofing-master" / "resources" / "anti_spoof_models"
    device_id: int = 0  # GPU ID hoặc -1 cho CPU
    threshold: float = 0.80 # Ngưỡng xác định là mặt thật (tối đa 1.0)

db_config     = DatabaseConfig()
ai_config     = AIConfig()
anti_spoof_config = AntiSpoofConfig()
camera_config = CameraConfig()
report_config = ReportConfig()
app_config    = AppConfig()