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
    det_size:         tuple = (640, 640)  # [Đã Nâng Cấp] Tăng kích thước lưới quét AI

    # Threshold 0.65 là "Điểm Vàng" theo công bố của InsightFace cho ảnh lấy từ Camera Thực tế
    recognition_threshold: float = float(os.getenv("AI_THRESHOLD", "0.65")) 
    min_enroll_photos:  int = 10
    max_enroll_photos:  int = 15
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
    width:  int = 1920
    height: int = 1080
    reconnect_delay_sec: int = 3
    max_reconnect_tries: int = 5

    # TĂNG KHOẢNG CÁCH XỬ LÝ: Skip khung hình để giảm tải CPU/GPU
    process_every_n_frames: int = int(os.getenv("CAM_PROCESS_N", "3"))

    @property
    def is_ip_camera(self) -> bool:
        return str(self.source).startswith("rtsp://") or \
               str(self.source).startswith("http://")

CAMERAS: list[dict] = [
    {"id": 1, "name": "Camera 1", "source": "rtsp://admin:a1234567@192.168.1.17:554/cam/realmonitor?channel=1&subtype=0", "floor": 1, "active": True},
    {"id": 2, "name": "Camera 2", "source": "rtsp://admin:a1234567@192.168.1.23:554/cam/realmonitor?channel=1&subtype=0", "floor": 1, "active": True},
    {"id": 3, "name": "Camera 3", "source": "rtsp://admin:a1234567@192.168.1.19:554/cam/realmonitor?channel=1&subtype=0", "floor": 1, "active": True},
    {"id": 4, "name": "Camera 4", "source": "rtsp://admin:a1234567@192.168.1.20:554/cam/realmonitor?channel=1&subtype=0", "floor": 1, "active": True},
    {"id": 5, "name": "Camera 5", "source": "rtsp://admin:a1234567@192.168.1.21:554/cam/realmonitor?channel=1&subtype=0", "floor": 1, "active": True},
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
    enabled:   bool = os.getenv("ENABLE_ANTISPOOF", "true").lower() == "true"
    model_dir: Path = BASE_DIR / "Silent-Face-Anti-Spoofing-master" / "resources" / "anti_spoof_models"
    device_id: int = 0  # GPU ID hoặc -1 cho CPU
    threshold: float = 0.80 # Ngưỡng xác định là mặt thật (tối đa 1.0)

# ─────────────────────────────────────────────
#  EDGE CONFIG — Cấu hình cho Mini PC
# ─────────────────────────────────────────────
@dataclass
class EdgeConfig:
    """Cấu hình khi chạy ở chế độ Edge (Mini PC)."""
    server_url:           str  = os.getenv("EDGE_SERVER_URL", "http://192.168.10.100:8000")
    api_key:              str  = os.getenv("EDGE_API_KEY", "faceattend_secret_2026")
    camera_id:            str  = os.getenv("EDGE_CAMERA_ID", "CAM_01")
    camera_source:        str  = os.getenv("EDGE_CAMERA_SOURCE", "0")  # "0" = webcam, "rtsp://..." = IP cam
    device_name:          str  = os.getenv("EDGE_DEVICE_NAME", "Edge Box 01")

    # Chu kỳ đồng bộ
    sync_interval_sec:    int  = int(os.getenv("EDGE_SYNC_INTERVAL", "30"))
    embedding_refresh_min: int = int(os.getenv("EDGE_EMBED_REFRESH", "10"))  # Reload embeddings mỗi N phút

    # Cooldown & Hiệu suất
    attendance_cooldown:  int  = int(os.getenv("EDGE_COOLDOWN", "60"))
    process_every_n:      int  = int(os.getenv("EDGE_PROCESS_N", "3"))

    # Hiển thị
    fullscreen:           bool = os.getenv("EDGE_FULLSCREEN", "true").lower() == "true"
    show_fps:             bool = os.getenv("EDGE_SHOW_FPS", "true").lower() == "true"
    auto_start:           bool = os.getenv("EDGE_AUTO_START", "true").lower() == "true"

    # Cấu hình danh sách camera (Mặc định cho Multi-cam)
    camera_list: list = field(default_factory=lambda: [
        {"id": "CAM_01", "name": "Camera 1", "source": "rtsp://admin:a1234567@192.168.1.17:554/cam/realmonitor?channel=1&subtype=0"},
        {"id": "CAM_02", "name": "Camera 2", "source": "rtsp://admin:a1234567@192.168.1.23:554/cam/realmonitor?channel=1&subtype=0"},
        {"id": "CAM_03", "name": "Camera 3", "source": "rtsp://admin:a1234567@192.168.1.19:554/cam/realmonitor?channel=1&subtype=0"},
        {"id": "CAM_04", "name": "Camera 4", "source": "rtsp://admin:a1234567@192.168.1.20:554/cam/realmonitor?channel=1&subtype=0"},
        {"id": "CAM_05", "name": "Camera 5", "source": "rtsp://admin:a1234567@192.168.1.21:554/cam/realmonitor?channel=1&subtype=0"},
    ])

db_config     = DatabaseConfig()
ai_config     = AIConfig()
anti_spoof_config = AntiSpoofConfig()
camera_config = CameraConfig()
report_config = ReportConfig()
app_config    = AppConfig()
edge_config   = EdgeConfig()