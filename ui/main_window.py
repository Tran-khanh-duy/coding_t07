"""
ui/main_window.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cửa sổ chính — điều phối toàn bộ ứng dụng.

Cấu trúc:
  ┌─────────┬──────────────────────────────┐
  │         │                              │
  │ Sidebar │     Page Stack (content)     │
  │  (220)  │                              │
  │         │                              │
  └─────────┴──────────────────────────────┘

Khởi động:
  1. Load model AI (thread riêng)
  2. Kết nối DB + load embeddings cache
  3. Hiển thị Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys
import threading
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout,
    QStackedWidget, QLabel, QVBoxLayout,
    QApplication, QMessageBox, QSplashScreen,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QFont, QIcon, QPixmap, QColor
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import app_config
from ui.styles.theme import Colors, apply_theme
from ui.widgets.sidebar import Sidebar
from ui.pages.dashboard_page import DashboardPage
from ui.pages.enroll_page import EnrollPage
from ui.pages.students_page import StudentsPage
from ui.pages.attendance_page import AttendancePage
from ui.pages.reports_page import ReportsPage
from utils.logger import setup_logger

# Setup logger
setup_logger(app_config.log_dir, app_config.log_level)


# ─────────────────────────────────────────────
#  Worker thread: khởi tạo nền
# ─────────────────────────────────────────────
class InitWorker(QThread):
    """
    Chạy các tác vụ khởi động nặng trong thread riêng
    để UI không bị freeze khi load.
    """
    progress      = pyqtSignal(str, int)   # (message, percent)
    model_ready   = pyqtSignal(bool)
    db_ready      = pyqtSignal(bool)
    cache_ready   = pyqtSignal(int)        # số embedding đã load
    init_done     = pyqtSignal()

    def run(self):
        # ── Bước 1: Kết nối DB ──
        self.progress.emit("Đang kết nối SQL Server...", 10)
        try:
            from database.connection import db
            ok = db.test_connection()
            self.db_ready.emit(ok)
            if ok:
                self.progress.emit("SQL Server: Đã kết nối ✅", 25)
            else:
                self.progress.emit("SQL Server: Không kết nối được ⚠️", 25)
        except Exception as e:
            logger.error(f"DB init error: {e}")
            self.db_ready.emit(False)
            self.progress.emit(f"SQL Server: Lỗi — {e}", 25)

        # ── Bước 2: Load embeddings cache ──
        self.progress.emit("Đang load embeddings từ DB...", 40)
        try:
            from services.embedding_cache_manager import cache_manager
            cache_manager.load()
            self.cache_ready.emit(cache_manager.size)
            self.progress.emit(
                f"Cache: {cache_manager.size} học viên đã load ✅", 55
            )
        except Exception as e:
            logger.error(f"Cache init error: {e}")
            self.cache_ready.emit(0)

        # ── Bước 3: Load model AI ──
        self.progress.emit("Đang load model AI buffalo_l... (có thể mất 1-2 phút lần đầu)", 60)
        try:
            from services.face_engine import face_engine
            ok = face_engine.load_model()
            self.model_ready.emit(ok)
            if ok:
                self.progress.emit("Model AI: Sẵn sàng ✅", 90)
            else:
                self.progress.emit("Model AI: Load thất bại ⚠️", 90)
        except Exception as e:
            logger.error(f"Model init error: {e}")
            self.model_ready.emit(False)
            self.progress.emit(f"Model AI: Lỗi — {e}", 90)

        self.progress.emit("Hoàn tất khởi động!", 100)
        self.init_done.emit()


# ─────────────────────────────────────────────
#  Placeholder page (chưa làm)
# ─────────────────────────────────────────────
class PlaceholderPage(QWidget):
    def __init__(self, icon: str, title: str, desc: str = "", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)

        icon_lbl = QLabel(icon)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_lbl.setStyleSheet("font-size: 56px;")
        layout.addWidget(icon_lbl)

        title_lbl = QLabel(title)
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_lbl.setStyleSheet(
            f"font-size: 20px; font-weight: 800; color: {Colors.TEXT};"
        )
        layout.addWidget(title_lbl)

        if desc:
            desc_lbl = QLabel(desc)
            desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            desc_lbl.setStyleSheet(f"font-size: 13px; color: {Colors.TEXT_DIM};")
            layout.addWidget(desc_lbl)

        badge = QLabel("🚧  Đang phát triển — Bước tiếp theo")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(f"""
            background: {Colors.ORANGE}18;
            color: {Colors.ORANGE};
            border: 1px solid {Colors.ORANGE}44;
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 12px;
            font-weight: 700;
            margin: 8px 40px;
        """)
        layout.addWidget(badge)


# ─────────────────────────────────────────────
#  Loading overlay
# ─────────────────────────────────────────────
class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setStyleSheet(f"background-color: {Colors.BG_DARK};")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)

        logo = QLabel("👁")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setStyleSheet("font-size: 64px;")
        layout.addWidget(logo)

        title = QLabel("FaceAttend")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"font-size: 26px; font-weight: 900; color: {Colors.TEXT}; letter-spacing: 2px;"
        )
        layout.addWidget(title)

        self._msg_lbl = QLabel("Đang khởi động...")
        self._msg_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._msg_lbl.setStyleSheet(
            f"font-size: 13px; color: {Colors.TEXT_DIM}; max-width: 400px;"
        )
        self._msg_lbl.setWordWrap(True)
        layout.addWidget(self._msg_lbl)

        # Progress bar thủ công (QLabel)
        self._bar_bg = QWidget()
        self._bar_bg.setFixedSize(360, 6)
        self._bar_bg.setStyleSheet(
            f"background: {Colors.BORDER}; border-radius: 3px;"
        )
        self._bar_fill = QWidget(self._bar_bg)
        self._bar_fill.setFixedHeight(6)
        self._bar_fill.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {Colors.CYAN_DIM},stop:1 {Colors.CYAN});"
            f"border-radius: 3px;"
        )
        self._bar_fill.setFixedWidth(0)
        layout.addWidget(self._bar_bg, 0, Qt.AlignmentFlag.AlignHCenter)

        self._pct_lbl = QLabel("0%")
        self._pct_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pct_lbl.setStyleSheet(
            f"font-size: 12px; color: {Colors.CYAN}; font-weight: 700;"
        )
        layout.addWidget(self._pct_lbl)

    def set_progress(self, message: str, percent: int):
        self._msg_lbl.setText(message)
        width = int(360 * percent / 100)
        self._bar_fill.setFixedWidth(width)
        self._pct_lbl.setText(f"{percent}%")


# ─────────────────────────────────────────────
#  MainWindow
# ─────────────────────────────────────────────
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self._model_ok  = False
        self._db_ok     = False
        self._cache_size= 0

        self._setup_window()
        self._build_layout()
        self._start_init()

    # ─── Setup ────────────────────────────────

    def _setup_window(self):
        self.setWindowTitle(app_config.app_name)
        self.resize(app_config.window_width, app_config.window_height)
        self.setMinimumSize(1024, 680)
        # Căn giữa màn hình
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width()  - app_config.window_width)  // 2
        y = (screen.height() - app_config.window_height) // 2
        self.move(x, y)

    def _build_layout(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ──
        self._sidebar = Sidebar()
        self._sidebar.page_changed.connect(self._on_page_changed)
        root.addWidget(self._sidebar)

        # ── Page stack ──
        self._stack = QStackedWidget()
        self._stack.setStyleSheet(f"background-color: {Colors.BG_DARK};")
        root.addWidget(self._stack, 1)

        # Tạo các trang
        self._dashboard  = DashboardPage()
        self._attendance = AttendancePage()
        self._enroll     = EnrollPage()
        self._students   = StudentsPage()
        self._reports    = ReportsPage()

        self._stack.addWidget(self._dashboard)                                        # 0
        self._stack.addWidget(self._attendance)                                       # 1
        self._stack.addWidget(self._enroll)                                           # 2
        self._stack.addWidget(self._students)                                         # 3
        self._stack.addWidget(self._reports)                                          # 4
        self._stack.addWidget(PlaceholderPage("⚙️","Cài Đặt","Cấu hình hệ thống")) # 5

        # Kết nối signal: Students → Enroll (truyền student_id)
        self._students.go_to_enroll.connect(self._navigate_to_enroll)

        # ── Loading overlay (đặt lên trên) ──
        self._loading = LoadingOverlay(central)
        self._loading.resize(app_config.window_width, app_config.window_height)
        self._loading.show()
        self._loading.raise_()

    # ─── Khởi động nền ────────────────────────

    def _start_init(self):
        self._worker = InitWorker()
        self._worker.progress.connect(self._on_init_progress)
        self._worker.model_ready.connect(self._on_model_ready)
        self._worker.db_ready.connect(self._on_db_ready)
        self._worker.cache_ready.connect(self._on_cache_ready)
        self._worker.init_done.connect(self._on_init_done)
        self._worker.start()

    def _on_init_progress(self, msg: str, pct: int):
        self._loading.set_progress(msg, pct)
        logger.info(f"Init [{pct}%]: {msg}")

    def _on_model_ready(self, ok: bool):
        self._model_ok = ok
        self._dashboard.update_system_status(
            "ai_model", ok,
            "buffalo_l — GPU ✅" if ok else "Load thất bại ❌"
        )

    def _on_db_ready(self, ok: bool):
        self._db_ok = ok
        self._sidebar.set_db_status(ok)
        self._dashboard.update_system_status(
            "database", ok,
            "SQL Server Express ✅" if ok else "Không kết nối được ❌"
        )
        if ok:
            self._dashboard.update_system_status("gpu", True, "NVIDIA 940MX ✅")

    def _on_cache_ready(self, size: int):
        self._cache_size = size
        self._dashboard.update_stats(enrolled=size)

    def _on_init_done(self):
        # Ẩn loading sau 0.5s
        QTimer.singleShot(500, self._hide_loading)
        self._load_dashboard_stats()

    def _hide_loading(self):
        self._loading.hide()
        logger.info("Ứng dụng đã sẵn sàng!")

        if not self._db_ok:
            QMessageBox.warning(
                self, "Cảnh báo kết nối DB",
                "Không thể kết nối SQL Server.\n\n"
                "Kiểm tra:\n"
                "  1. SQL Server Express đang chạy\n"
                "  2. Server name trong config.py\n"
                "  3. Database FaceAttendanceDB đã tạo chưa\n\n"
                "Chạy scripts/create_database.sql để tạo DB."
            )

    def _load_dashboard_stats(self):
        """Load số liệu thống kê cho dashboard."""
        if not self._db_ok:
            return
        try:
            from database.repositories import student_repo, class_repo
            from services.camera_manager import camera_manager
            students = student_repo.get_all()
            enrolled = sum(1 for s in students if s.face_enrolled)
            self._dashboard.update_stats(
                students=len(students),
                enrolled=enrolled,
                cameras=camera_manager.get_connected_count(),
            )
            self._dashboard.update_system_status("camera", True, "Sẵn sàng")
        except Exception as e:
            logger.error(f"Load stats error: {e}")

    # ─── Navigation ───────────────────────────

    def _on_page_changed(self, page_id: int):
        self._stack.setCurrentIndex(page_id)
        logger.debug(f"Chuyển trang: {page_id}")

    def _navigate_to_enroll(self, student_id: int):
        """
        Được gọi khi Students → click 'Đăng ký mặt' / 'Cập nhật' / 'Thêm học viên'.
        student_id = -1  → form trống (thêm mới)
        student_id > 0   → điền sẵn thông tin học viên
        """
        try:
            if student_id > 0:
                self._enroll.load_student(student_id)
            else:
                self._enroll._reset_form()   # Thêm mới → form trống sạch
        except Exception as e:
            logger.error(f"_navigate_to_enroll error: {e}")
        self._sidebar._on_nav_click(Sidebar.PAGE_ENROLL)

    # ─── Đóng ứng dụng ────────────────────────

    def closeEvent(self, event):
        logger.info("Đóng ứng dụng...")
        try:
            from services.camera_manager import camera_manager
            camera_manager.stop_all()
        except Exception:
            pass
        try:
            from services.attendance_service import attendance_service
            if attendance_service.is_active:
                attendance_service.end_session()
        except Exception:
            pass
        event.accept()


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
def run():
    app = QApplication(sys.argv)
    app.setApplicationName("FaceAttend")
    app.setApplicationVersion("1.0.0")
    apply_theme(app)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()