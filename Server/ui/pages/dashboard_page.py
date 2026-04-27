"""
ui/pages/dashboard_page.py
"""
import subprocess
import platform
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QGridLayout, QScrollArea, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QFont, QColor, QPixmap, QPainter, QPen
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.styles.theme import Colors
from config import db_config, ai_config, CAMERAS
from services.face_engine import face_engine
from services.camera_manager import camera_manager, CameraStatus


def get_gpu_name():
    try:
        output = subprocess.check_output("wmic path win32_VideoController get name", shell=True, text=True)
        lines = [line.strip() for line in output.split('\n') if line.strip() and "Name" not in line]
        if lines:
            return lines[0]
    except Exception:
        pass
    return platform.processor()


class CircularProgress(QWidget):
    """Biểu đồ vòng tròn thể hiện tỷ lệ phần trăm."""
    def __init__(self, color_str, parent=None):
        super().__init__(parent)
        self.setFixedSize(140, 140)
        self.value = 0
        self.max_value = 100
        self.color = QColor(color_str)
        self.bg_color = QColor("#EEF2FF")

    def set_value(self, val, max_val=100):
        self.value = int(val)
        self.max_value = int(max_val) if max_val > 0 else 1
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = QRectF(15, 15, 110, 110)
        
        # Nền vòng tròn
        pen_bg = QPen(self.bg_color)
        pen_bg.setWidth(14)
        pen_bg.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_bg)
        painter.drawArc(rect, 0, 360 * 16)
        
        # Vòng tròn giá trị
        pen_val = QPen(self.color)
        pen_val.setWidth(14)
        pen_val.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_val)
        
        span_angle = int((self.value / self.max_value) * 360 * 16)
        painter.drawArc(rect, 90 * 16, -span_angle)
        
        # Chữ ở giữa
        painter.setPen(QColor("#0F172A"))
        font = QFont("Segoe UI", 20, QFont.Weight.Black)
        painter.setFont(font)
        percentage = int((self.value / self.max_value) * 100) if self.max_value > 0 else 0
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{percentage}%")


class MetricCard(QFrame):
    """Thẻ hiển thị Số liệu chính (KPI) trên cùng."""
    def __init__(self, title, value, icon, color):
        super().__init__()
        self.setStyleSheet(f"""
            QFrame {{
                background: white; border-radius: 16px; border: 1px solid #E2E8F0;
            }}
            QFrame:hover {{ border: 1px solid {color}80; }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        
        left = QVBoxLayout()
        left.setSpacing(4)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-size: 13px; font-weight: 800; color: #64748B; border: none; letter-spacing: 0.5px;")
        
        self.val_lbl = QLabel(value)
        self.val_lbl.setStyleSheet(f"font-size: 38px; font-weight: 900; color: #0F172A; border: none;")
        
        left.addWidget(title_lbl)
        left.addWidget(self.val_lbl)
        left.addStretch()
        
        layout.addLayout(left)
        layout.addStretch()
        
        icon_lbl = QLabel(icon)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_lbl.setFixedSize(64, 64)
        icon_lbl.setStyleSheet(f"font-size: 30px; background: {color}15; color: {color}; border-radius: 32px; border: none;")
        layout.addWidget(icon_lbl)

    def set_value(self, val):
        self.val_lbl.setText(str(val))


class StatusTile(QFrame):
    """Thẻ trạng thái hệ thống nhỏ gọn ở dưới cùng."""
    def __init__(self, title, icon, color):
        super().__init__()
        self.setStyleSheet("""
            QFrame { background: white; border-radius: 12px; border: 1px solid #E2E8F0; }
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        
        ic = QLabel(icon)
        ic.setFixedSize(40, 40)
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ic.setStyleSheet(f"font-size: 20px; background: {color}15; border-radius: 12px; border: none;")
        layout.addWidget(ic)
        
        texts = QVBoxLayout()
        texts.setSpacing(2)
        t = QLabel(title)
        t.setStyleSheet("font-size: 12px; font-weight: 700; color: #64748B; border: none;")
        self.val = QLabel("Đang tải...")
        self.val.setStyleSheet("font-size: 14px; font-weight: 800; color: #0F172A; border: none;")
        texts.addWidget(t)
        texts.addWidget(self.val)
        layout.addLayout(texts)
        layout.addStretch()
        
        self.indicator = QLabel()
        self.indicator.setFixedSize(12, 12)
        self.indicator.setStyleSheet(f"background: #CBD5E1; border-radius: 6px; border: none;")
        layout.addWidget(self.indicator)
        
    def set_status(self, text, is_ok):
        self.val.setText(text)
        color = Colors.GREEN if is_ok else Colors.RED
        self.indicator.setStyleSheet(f"background: {color}; border-radius: 6px; border: none;")


class CameraPill(QFrame):
    """Hiển thị trạng thái của từng Camera."""
    def __init__(self, cam_id, name):
        super().__init__()
        self.cam_id = cam_id
        self.setStyleSheet("""
            QFrame { background: #F8FAFC; border-radius: 8px; border: 1px solid #E2E8F0; }
        """)
        self.setFixedHeight(50)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        
        self.icon = QLabel("📷")
        self.icon.setStyleSheet("border: none; background: transparent; font-size: 16px;")
        
        self.name_lbl = QLabel(name)
        self.name_lbl.setStyleSheet("font-size: 13px; font-weight: 700; color: #334155; border: none; background: transparent;")
        
        self.status_lbl = QLabel("Offline")
        self.status_lbl.setStyleSheet("font-size: 12px; font-weight: 800; color: #EF4444; border: none; background: transparent;")
        
        layout.addWidget(self.icon)
        layout.addWidget(self.name_lbl)
        layout.addStretch()
        layout.addWidget(self.status_lbl)

    def update_status(self, is_online):
        if is_online:
            self.status_lbl.setText("Online")
            self.status_lbl.setStyleSheet(f"font-size: 12px; font-weight: 800; color: {Colors.GREEN}; border: none; background: transparent;")
        else:
            self.status_lbl.setText("Offline")
            self.status_lbl.setStyleSheet("font-size: 12px; font-weight: 800; color: #EF4444; border: none; background: transparent;")


class DashboardPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gpu_name = get_gpu_name()
        self.setStyleSheet("background-color: #F1F5F9;")  # Màu xám siêu nhạt chuẩn Dashboard
        
        self._students_total = 0
        self._present_count = 0
        self._absent_count = 0

        self._setup_ui()

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)
        self._update_clock()
        
        self._sys_timer = QTimer(self)
        self._sys_timer.timeout.connect(self._check_system_realtime)
        self._sys_timer.start(3000)

    def _setup_ui(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: transparent;")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        main = QVBoxLayout(container)
        main.setContentsMargins(40, 40, 40, 40)
        main.setSpacing(24)

        # ── HEADER ──
        header = QHBoxLayout()
        
        self._logo_lbl = QLabel()
        logo_path = Path(__file__).parent.parent / "pictures" / "T07-2025 (DOITEN) (1).png"
        pixmap = QPixmap(str(logo_path))
        if not pixmap.isNull():
            pixmap = pixmap.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self._logo_lbl.setPixmap(pixmap)
        else:
            self._logo_lbl.setText("ACT")
            self._logo_lbl.setFixedSize(80, 80)
            self._logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._logo_lbl.setStyleSheet(f"background: {Colors.CYAN}; color: white; font-size: 28px; font-weight: 900; border-radius: 40px;")
        
        header.addWidget(self._logo_lbl)
        header.addSpacing(15)

        title_col = QVBoxLayout()
        school_lbl = QLabel("HỌC VIỆN KỸ THUẬT VÀ CÔNG NGHỆ AN NINH")
        school_lbl.setStyleSheet(f"font-size: 15px; font-weight: 900; color: {Colors.CYAN};")
        sys_lbl = QLabel("PHÒNG QUẢN LÝ HỌC VIÊN")
        sys_lbl.setStyleSheet(f"font-size: 24px; font-weight: 900; color: #0F172A; letter-spacing: -0.5px;")
        title_col.addWidget(school_lbl)
        title_col.addWidget(sys_lbl)
        header.addLayout(title_col)
        header.addStretch()

        time_col = QVBoxLayout()
        self._clock_lbl = QLabel()
        self._clock_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._clock_lbl.setStyleSheet(f"font-size: 32px; font-weight: 900; color: #0F172A; font-family: monospace;")
        self._date_lbl = QLabel()
        self._date_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._date_lbl.setStyleSheet("font-size: 14px; font-weight: 600; color: #64748B;")
        time_col.addWidget(self._clock_lbl)
        time_col.addWidget(self._date_lbl)
        header.addLayout(time_col)
        
        main.addLayout(header)

        # ── KPI ROW (TOP) ──
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(24)
        
        self.kpi_total = MetricCard("TỔNG HỌC VIÊN", "0", "👥", Colors.CYAN)
        self.kpi_present = MetricCard("CÓ MẶT HÔM NAY", "0", "✨", Colors.GREEN)
        self.kpi_absent = MetricCard("VẮNG MẶT", "0", "⚠️", Colors.ORANGE)
        
        kpi_row.addWidget(self.kpi_total)
        kpi_row.addWidget(self.kpi_present)
        kpi_row.addWidget(self.kpi_absent)
        main.addLayout(kpi_row)

        # ── MIDDLE ROW: Chart & Cameras ──
        mid_row = QHBoxLayout()
        mid_row.setSpacing(24)

        # L: Tỉ lệ điểm danh (Chart Panel)
        chart_panel = QFrame()
        chart_panel.setStyleSheet("background: white; border-radius: 16px; border: 1px solid #E2E8F0;")
        chart_layout = QVBoxLayout(chart_panel)
        chart_layout.setContentsMargins(30, 24, 30, 30)
        
        c_title = QLabel("TỶ LỆ ĐIỂM DANH")
        c_title.setStyleSheet("font-size: 16px; font-weight: 900; color: #1E293B; border: none;")
        chart_layout.addWidget(c_title)
        
        chart_body = QHBoxLayout()
        self.donut = CircularProgress(Colors.CYAN)
        chart_body.addStretch()
        chart_body.addWidget(self.donut)
        chart_body.addSpacing(40)
        
        bars_layout = QVBoxLayout()
        bars_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        
        def make_bar_row(lbl_text, color):
            row = QVBoxLayout()
            row.setSpacing(4)
            l = QLabel(lbl_text)
            l.setStyleSheet("font-size: 13px; font-weight: 700; color: #475569; border: none;")
            bar = QProgressBar()
            bar.setFixedHeight(8)
            bar.setTextVisible(False)
            bar.setStyleSheet(f"""
                QProgressBar {{ background: #F1F5F9; border-radius: 4px; border: none; }}
                QProgressBar::chunk {{ background: {color}; border-radius: 4px; }}
            """)
            row.addWidget(l)
            row.addWidget(bar)
            return row, bar

        r1, self.bar_present = make_bar_row("Đã điểm danh (Có mặt)", Colors.GREEN)
        r2, self.bar_absent = make_bar_row("Chưa nhận diện được", Colors.ORANGE)
        
        bars_layout.addLayout(r1)
        bars_layout.addSpacing(15)
        bars_layout.addLayout(r2)
        
        chart_body.addLayout(bars_layout)
        chart_body.addStretch()
        
        chart_layout.addLayout(chart_body)
        mid_row.addWidget(chart_panel, 6)

        # R: Camera Live Status
        cam_panel = QFrame()
        cam_panel.setStyleSheet("background: white; border-radius: 16px; border: 1px solid #E2E8F0;")
        cam_layout = QVBoxLayout(cam_panel)
        cam_layout.setContentsMargins(24, 24, 24, 24)
        
        ct = QLabel("GIÁM SÁT CAMERA")
        ct.setStyleSheet("font-size: 16px; font-weight: 900; color: #1E293B; border: none;")
        cam_layout.addWidget(ct)
        cam_layout.addSpacing(10)
        
        self.cam_pills = {}
        # Render up to 5 cameras
        for cam in CAMERAS[:5]:
            pill = CameraPill(cam["id"], f"{cam['name']} (Tầng {cam.get('floor', '?')})")
            cam_layout.addWidget(pill)
            self.cam_pills[cam["id"]] = pill
            
        cam_layout.addStretch()
        mid_row.addWidget(cam_panel, 4)

        main.addLayout(mid_row)

        # ── BOTTOM ROW: System Health ──
        health_row = QHBoxLayout()
        health_row.setSpacing(20)
        
        self.health_ai = StatusTile("AI Engine", "🧠", Colors.CYAN)
        server_name = db_config.server.split('\\')[-1] 
        self.health_db = StatusTile("Database", "🗄️", Colors.PURPLE)
        self.health_gpu = StatusTile("GPU Compute", "⚡", Colors.ORANGE)
        self.health_sys = StatusTile("Server Status", "🖥️", Colors.GREEN)
        
        health_row.addWidget(self.health_ai)
        health_row.addWidget(self.health_db)
        health_row.addWidget(self.health_gpu)
        health_row.addWidget(self.health_sys)
        
        main.addLayout(health_row)
        main.addStretch()

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _update_clock(self):
        now = datetime.now()
        self._clock_lbl.setText(now.strftime("%H:%M:%S"))
        self._date_lbl.setText(now.strftime("%d/%m/%Y"))

    def _check_system_realtime(self):
        # AI
        if face_engine.is_ready:
            self.health_ai.set_status(ai_config.model_name, True)
        else:
            self.health_ai.set_status("Loading...", False)
            
        # DB
        server_name = db_config.server.split('\\')[-1]
        self.health_db.set_status(server_name, True)
        
        # GPU
        self.health_gpu.set_status(self.gpu_name, True)
        
        # SYS
        self.health_sys.set_status("Running", True)
        
        # Cameras
        for cam_id, pill in self.cam_pills.items():
            status = camera_manager.get_status(cam_id)
            is_online = (status == CameraStatus.CONNECTED)
            pill.update_status(is_online)

    def update_system_status(self, key: str, ok: bool, text: str = "", custom_color: str = None):
        """Hỗ trợ tương thích ngược cho main_window.py gọi"""
        if key == "database":
            self.health_db.set_status(text, ok)
        elif key == "ai_model":
            self.health_ai.set_status(text, ok)
        elif key == "gpu":
            self.health_gpu.set_status(text, ok)

    def update_stats(self, **kwargs):
        if "students" in kwargs:
            self.kpi_total.set_value(kwargs["students"])
            self._students_total = kwargs["students"]
            
        if "present" in kwargs:
            self.kpi_present.set_value(kwargs["present"])
            self._present_count = kwargs["present"]
            
        if "absent" in kwargs:
            self.kpi_absent.set_value(kwargs["absent"])
            self._absent_count = kwargs["absent"]
            
        # Cập nhật Biểu đồ
        if self._students_total > 0:
            self.donut.set_value(self._present_count, self._students_total)
            self.bar_present.setRange(0, self._students_total)
            self.bar_present.setValue(self._present_count)
            
            self.bar_absent.setRange(0, self._students_total)
            self.bar_absent.setValue(self._absent_count)