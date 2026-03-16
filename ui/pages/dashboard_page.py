"""
ui/pages/dashboard_page.py
Trang Dashboard — tổng quan hệ thống.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QGridLayout, QScrollArea,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ui.styles.theme import Colors, card_style


class StatCard(QWidget):
    """Card hiển thị 1 số liệu thống kê."""
    def __init__(self, icon: str, title: str, value: str, color: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            background-color: {Colors.BG_CARD};
            border: 1px solid {color}30;
            border-radius: 12px;
            padding: 0px;
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(8)

        top = QHBoxLayout()
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet(f"font-size: 24px; background: transparent;")
        top.addWidget(icon_lbl)
        top.addStretch()
        layout.addLayout(top)

        self._value_lbl = QLabel(value)
        self._value_lbl.setStyleSheet(
            f"font-size: 28px; font-weight: 900; color: {color}; background: transparent;"
        )
        layout.addWidget(self._value_lbl)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"font-size: 12px; color: {Colors.TEXT_DIM}; background: transparent;"
        )
        layout.addWidget(title_lbl)

        # Accent bottom border
        accent = QFrame()
        accent.setFixedHeight(3)
        accent.setStyleSheet(f"background: {color}; border-radius: 2px;")
        layout.addWidget(accent)

    def set_value(self, value: str):
        self._value_lbl.setText(value)


class DashboardPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

        # Cập nhật đồng hồ mỗi giây
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)
        self._update_clock()

    def _setup_ui(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")

        container = QWidget()
        main = QVBoxLayout(container)
        main.setContentsMargins(28, 24, 28, 24)
        main.setSpacing(20)

        # ── Header ──
        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(4)

        title = QLabel("Dashboard")
        title.setProperty("class", "title")
        title.setStyleSheet(f"font-size: 22px; font-weight: 900; color: {Colors.TEXT};")

        self._date_lbl = QLabel()
        self._date_lbl.setStyleSheet(f"font-size: 13px; color: {Colors.TEXT_DIM};")
        title_col.addWidget(title)
        title_col.addWidget(self._date_lbl)
        header.addLayout(title_col)
        header.addStretch()

        self._clock_lbl = QLabel()
        self._clock_lbl.setStyleSheet(
            f"font-size: 28px; font-weight: 900; color: {Colors.CYAN}; letter-spacing: 2px;"
        )
        header.addWidget(self._clock_lbl)
        main.addLayout(header)

        # ── Stat Cards ──
        grid = QGridLayout()
        grid.setSpacing(14)

        self._cards = {
            "students":  StatCard("👥", "Tổng học viên",     "—", Colors.CYAN),
            "enrolled":  StatCard("✅", "Đã đăng ký mặt",    "—", Colors.GREEN),
            "sessions":  StatCard("📅", "Buổi hôm nay",       "—", Colors.ORANGE),
            "present":   StatCard("📷", "Có mặt hôm nay",    "—", Colors.GREEN),
            "absent":    StatCard("⚠️", "Vắng hôm nay",       "—", Colors.RED),
            "cameras":   StatCard("🎥", "Camera hoạt động",  "—", Colors.PURPLE),
        }
        positions = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
        for (card, widget), (r, c) in zip(self._cards.items(), positions):
            grid.addWidget(widget, r, c)
        main.addLayout(grid)

        # ── Trạng thái hệ thống ──
        sys_card = QWidget()
        sys_card.setStyleSheet(card_style(Colors.BORDER))
        sys_layout = QVBoxLayout(sys_card)
        sys_layout.setSpacing(12)

        sys_title = QLabel("TRẠNG THÁI HỆ THỐNG")
        sys_title.setStyleSheet(
            f"font-size: 11px; font-weight: 700; color: {Colors.TEXT_DIM}; letter-spacing: 1.5px;"
        )
        sys_layout.addWidget(sys_title)

        self._status_rows: dict[str, QLabel] = {}
        status_items = [
            ("ai_model",  "🧠", "Model AI (buffalo_l)"),
            ("database",  "🗄️", "SQL Server"),
            ("camera",    "📷", "Camera"),
            ("gpu",       "🎮", "GPU (NVIDIA 940MX)"),
        ]
        for key, icon, label in status_items:
            row = QHBoxLayout()
            row.setSpacing(10)
            row_lbl = QLabel(f"{icon}  {label}")
            row_lbl.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 13px;")
            status_lbl = QLabel("● Chưa kiểm tra")
            status_lbl.setStyleSheet(f"color: {Colors.TEXT_DARK}; font-size: 13px;")
            self._status_rows[key] = status_lbl
            row.addWidget(row_lbl)
            row.addStretch()
            row.addWidget(status_lbl)
            sys_layout.addLayout(row)

        main.addWidget(sys_card)

        # ── Hướng dẫn nhanh ──
        guide_card = QWidget()
        guide_card.setStyleSheet(card_style(Colors.CYAN + "30"))
        guide_layout = QVBoxLayout(guide_card)
        guide_title = QLabel("🚀  HƯỚNG DẪN NHANH")
        guide_title.setStyleSheet(
            f"font-size: 11px; font-weight: 700; color: {Colors.CYAN}; letter-spacing: 1.5px;"
        )
        guide_layout.addWidget(guide_title)

        steps = [
            ("1", "Đăng ký học viên", "Vào tab 'Đăng Ký HV' → Nhập thông tin → Chụp 10 ảnh khuôn mặt"),
            ("2", "Tạo buổi điểm danh", "Vào tab 'Điểm Danh' → Chọn lớp + môn học → Bắt đầu"),
            ("3", "Điểm danh", "Học viên lần lượt đưa mặt vào camera → Hệ thống tự nhận diện"),
            ("4", "Xuất báo cáo", "Vào tab 'Báo Cáo' → Chọn buổi → Xuất Excel/PDF"),
        ]
        for num, title, desc in steps:
            step_row = QHBoxLayout()
            step_row.setSpacing(12)
            num_lbl = QLabel(num)
            num_lbl.setFixedSize(24, 24)
            num_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            num_lbl.setStyleSheet(f"""
                background: {Colors.CYAN}22;
                color: {Colors.CYAN};
                border: 1px solid {Colors.CYAN}44;
                border-radius: 12px;
                font-weight: 700;
                font-size: 12px;
            """)
            text_col = QVBoxLayout()
            text_col.setSpacing(2)
            t_lbl = QLabel(title)
            t_lbl.setStyleSheet(f"color: {Colors.TEXT}; font-weight: 700; font-size: 13px;")
            d_lbl = QLabel(desc)
            d_lbl.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 12px;")
            text_col.addWidget(t_lbl)
            text_col.addWidget(d_lbl)
            step_row.addWidget(num_lbl, 0, Qt.AlignmentFlag.AlignTop)
            step_row.addLayout(text_col)
            guide_layout.addLayout(step_row)

        main.addWidget(guide_card)
        main.addStretch()

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _update_clock(self):
        now = datetime.now()
        self._clock_lbl.setText(now.strftime("%H:%M:%S"))
        days = ["Thứ Hai","Thứ Ba","Thứ Tư","Thứ Năm","Thứ Sáu","Thứ Bảy","Chủ Nhật"]
        self._date_lbl.setText(f"{days[now.weekday()]}, {now.strftime('%d/%m/%Y')}")

    def update_system_status(self, key: str, ok: bool, text: str = ""):
        if key in self._status_rows:
            lbl = self._status_rows[key]
            color = Colors.GREEN if ok else Colors.RED
            msg = text or ("Hoạt động" if ok else "Lỗi")
            lbl.setText(f"● {msg}")
            lbl.setStyleSheet(f"color: {color}; font-size: 13px;")

    def update_stats(self, **kwargs):
        mapping = {
            "students": "students", "enrolled": "enrolled",
            "sessions": "sessions", "present":  "present",
            "absent":   "absent",   "cameras":  "cameras",
        }
        for key, val in kwargs.items():
            if key in mapping and mapping[key] in self._cards:
                self._cards[mapping[key]].set_value(str(val))