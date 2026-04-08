"""
ui/pages/dashboard_page.py
"""
import subprocess
import platform
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QGridLayout, QScrollArea, QComboBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.styles.theme import Colors, card_style
from config import db_config, ai_config, CAMERAS
from services.face_engine import face_engine
from services.camera_manager import camera_manager, CameraStatus


def get_gpu_name():
    """Hàm lấy tên Card đồ hoạ thực tế của Windows bằng lệnh CMD/WMI"""
    try:
        output = subprocess.check_output("wmic path win32_VideoController get name", shell=True, text=True)
        lines = [line.strip() for line in output.split('\n') if line.strip() and "Name" not in line]
        if lines:
            return lines[0]
    except Exception:
        pass
    # Fallback nếu không quét được
    return platform.processor()


class StatCard(QWidget):
    """Card hiển thị 1 số liệu thống kê (Đã nâng cấp chống mất chữ, hỗ trợ subtext)."""
    def __init__(self, icon: str, title: str, value: str, subtext: str, color: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {Colors.BG_CARD};
                border: 1px solid {color}30;
                border-radius: 12px;
            }}
        """)
        
        # Đảm bảo layout co giãn vừa vặn chữ
        self.setMinimumHeight(130)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(6)

        top = QHBoxLayout()
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet(f"font-size: 26px; border: none; background: transparent;")
        top.addWidget(icon_lbl)
        top.addStretch()
        layout.addLayout(top)

        self._value_lbl = QLabel(value)
        self._value_lbl.setStyleSheet(
            f"font-size: 32px; font-weight: 900; color: {color}; border: none; background: transparent;"
        )
        layout.addWidget(self._value_lbl)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {Colors.TEXT_DIM}; border: none; background: transparent;"
        )
        layout.addWidget(title_lbl)

        self._sub_lbl = QLabel(subtext)
        self._sub_lbl.setStyleSheet(
            f"font-size: 12px; color: {Colors.TEXT_DARK}; border: none; background: transparent;"
        )
        self._sub_lbl.setWordWrap(True)
        layout.addWidget(self._sub_lbl)

        # Accent bottom border
        accent = QFrame()
        accent.setFixedHeight(4)
        accent.setStyleSheet(f"background: {color}; border-radius: 2px; border: none;")
        layout.addWidget(accent)

    def set_value(self, value: str, subtext: str = None):
        self._value_lbl.setText(value)
        if subtext is not None:
            self._sub_lbl.setText(subtext)


class DashboardPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gpu_name = get_gpu_name()
        self._setup_ui()

        # Timer cập nhật đồng hồ 1s / lần
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)
        self._update_clock()
        
        # Timer cập nhật trạng thái hệ thống 3s / lần
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
        main.setContentsMargins(30, 26, 30, 26)
        main.setSpacing(24)

        # ── Header ──
        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(6)

        title = QLabel("Dashboard")
        title.setProperty("class", "title")
        title.setStyleSheet(f"font-size: 26px; font-weight: 900; color: {Colors.TEXT};")

        self._date_lbl = QLabel()
        self._date_lbl.setStyleSheet(f"font-size: 14px; color: {Colors.TEXT_DIM};")
        title_col.addWidget(title)
        title_col.addWidget(self._date_lbl)
        header.addLayout(title_col)
        header.addStretch()

        self._clock_lbl = QLabel()
        self._clock_lbl.setStyleSheet(
            f"font-size: 32px; font-weight: 900; color: {Colors.CYAN}; letter-spacing: 2px;"
        )
        header.addWidget(self._clock_lbl)
        main.addLayout(header)

        # ── Stat Cards (Rút gọn 4 thẻ) ──
        grid = QGridLayout()
        grid.setSpacing(18)

        self._cards = {
            "students":  StatCard("👥", "Tổng học viên", "—", "Đã lưu trong CSDL", Colors.CYAN),
            "present":   StatCard("✅", "Có mặt", "—", "Phiên gần nhất: Chưa có", Colors.GREEN),
            "absent":    StatCard("⚠️", "Vắng mặt", "—", "Phiên gần nhất: Chưa có", Colors.ORANGE),
            "cameras":   StatCard("🎥", "Camera hoạt động", "—", "Tổng luồng stream", Colors.PURPLE),
        }
        
        # Sắp xếp 2x2
        grid.addWidget(self._cards["students"], 0, 0)
        grid.addWidget(self._cards["present"], 0, 1)
        grid.addWidget(self._cards["absent"], 1, 0)
        grid.addWidget(self._cards["cameras"], 1, 1)
        main.addLayout(grid)

        # ── Trạng thái hệ thống (Hiển thị Realtime) ──
        sys_card = QWidget()
        sys_card.setStyleSheet(card_style(Colors.BORDER, radius=12))
        sys_layout = QVBoxLayout(sys_card)
        sys_layout.setContentsMargins(24, 20, 24, 20)
        sys_layout.setSpacing(16)

        sys_title = QLabel("TRẠNG THÁI HỆ THỐNG")
        sys_title.setStyleSheet(
            f"font-size: 13px; font-weight: 800; color: {Colors.TEXT_DIM}; letter-spacing: 1.5px; border: none;"
        )
        sys_layout.addWidget(sys_title)

        self._status_rows = {}
        
        # Dòng 1: AI Model
        self._add_status_row(sys_layout, "ai_model", "🧠", f"Mô hình AI ({ai_config.model_name})")
        
        # Dòng 2: Database Server
        # Tách việc xử lý tên server ra ngoài
        server_name = db_config.server.split('\\')[-1] 

        # Sau đó mới đưa vào f-string
        db_info = f"SQL Server ({server_name} - {db_config.database})"
        self._add_status_row(sys_layout, "database", "🗄️", db_info)
        
        # Dòng 3: Hardware
        hw_info = f"Phần cứng ({self.gpu_name})"
        self._add_status_row(sys_layout, "gpu", "🎮", hw_info)

        # Dòng 4: Camera (ComboBox lựa chọn)
        cam_row = QHBoxLayout()
        cam_row.setSpacing(12)
        cam_icon_lbl = QLabel("📷")
        cam_icon_lbl.setStyleSheet(f"font-size: 16px; border: none; background: transparent;")
        
        self._cam_combo = QComboBox()
        self._cam_combo.setMinimumWidth(250)
        for cam in CAMERAS:
            self._cam_combo.addItem(f"{cam['name']} - Tầng {cam.get('floor', 1)}", cam['id'])
        self._cam_combo.currentIndexChanged.connect(self._check_system_realtime)

        self._status_rows["camera"] = QLabel("● Chưa kiểm tra")
        self._status_rows["camera"].setStyleSheet(f"color: {Colors.TEXT_DARK}; font-size: 14px; font-weight: 600; border: none;")
        
        cam_row.addWidget(cam_icon_lbl)
        cam_row.addWidget(self._cam_combo)
        cam_row.addStretch()
        cam_row.addWidget(self._status_rows["camera"])
        sys_layout.addLayout(cam_row)

        main.addWidget(sys_card)
        main.addStretch()

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _add_status_row(self, parent_layout, key: str, icon: str, label: str):
        row = QHBoxLayout()
        row.setSpacing(10)
        row_lbl = QLabel(f"{icon}  {label}")
        row_lbl.setStyleSheet(f"color: {Colors.TEXT}; font-size: 14px; border: none; background: transparent; font-weight: 500;")
        
        status_lbl = QLabel("● Đang kiểm tra...")
        status_lbl.setStyleSheet(f"color: {Colors.ORANGE}; font-size: 14px; font-weight: 600; border: none; background: transparent;")
        
        self._status_rows[key] = status_lbl
        row.addWidget(row_lbl)
        row.addStretch()
        row.addWidget(status_lbl)
        parent_layout.addLayout(row)

    def _update_clock(self):
        now = datetime.now()
        self._clock_lbl.setText(now.strftime("%H:%M:%S"))
        days = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
        self._date_lbl.setText(f"{days[now.weekday()]}, {now.strftime('%d/%m/%Y')}")

    def _check_system_realtime(self):
        """Kiểm tra độ sẵn sàng của AI, Camera và xuất trạng thái (Realtime)"""
        # 1. AI Model
        if face_engine.is_ready:
            self.update_system_status("ai_model", True, "Sẵn sàng (Đã nạp vRAM)")
        else:
            self.update_system_status("ai_model", False, "Đang tải hoặc Lỗi")
            
        # 2. Hardware (Luôn sẵn sàng sau khi quét)
        self.update_system_status("gpu", True, "Hoạt động ổn định")
        
        # 3. Camera đang chọn trong Combo Box
        selected_cam_id = self._cam_combo.currentData()
        if selected_cam_id is not None:
            status = camera_manager.get_status(selected_cam_id)
            if status == CameraStatus.CONNECTED:
                self.update_system_status("camera", True, "Đang stream mượt mà")
            elif status == CameraStatus.CONNECTING:
                self.update_system_status("camera", False, "Đang kết nối lại...", Colors.ORANGE)
            else:
                self.update_system_status("camera", False, "Mất tín hiệu (Offline)")

    def update_system_status(self, key: str, ok: bool, text: str = "", custom_color: str = None):
        if key in self._status_rows:
            lbl = self._status_rows[key]
            if custom_color:
                color = custom_color
            else:
                color = Colors.GREEN if ok else Colors.RED
            lbl.setText(f"● {text}")
            lbl.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: 600; border: none;")

    def update_stats(self, **kwargs):
        """
        Nhận giá trị động từ App.
        Ví dụ: update_stats(students=150, present=45, absent=2, last_session_time="10/10/2026 08:00")
        """
        if "students" in kwargs:
            self._cards["students"].set_value(str(kwargs["students"]))
            
        if "cameras" in kwargs:
            self._cards["cameras"].set_value(f"{kwargs['cameras']} / {len(CAMERAS)}")

        last_time = kwargs.get("last_session_time", "Chưa có dữ liệu")
        subtext = f"Phiên gần nhất: {last_time}"
        
        if "present" in kwargs:
            self._cards["present"].set_value(str(kwargs["present"]), subtext)
            
        if "absent" in kwargs:
            self._cards["absent"].set_value(str(kwargs["absent"]), subtext)