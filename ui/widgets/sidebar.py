"""
ui/widgets/sidebar.py
Thanh điều hướng bên trái — Light theme, Windows 11 Fluent style.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ui.styles.theme import Colors


class NavButton(QPushButton):
    def __init__(self, icon_text: str, label: str, page_id: int):
        super().__init__()
        self.page_id = page_id
        self._active = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 0, 16, 0)
        layout.setSpacing(11)

        self._icon_lbl = QLabel(icon_text)
        self._icon_lbl.setFixedWidth(22)
        self._icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_lbl.setStyleSheet("font-size: 17px; background: transparent;")

        self._text_lbl = QLabel(label)

        layout.addWidget(self._icon_lbl)
        layout.addWidget(self._text_lbl)
        layout.addStretch()

        self.setFixedHeight(44)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh_style()

    def set_active(self, active: bool):
        self._active = active
        self._refresh_style()

    def _refresh_style(self):
        if self._active:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Colors.BG_SELECTED};
                    border: none;
                    border-left: 3px solid {Colors.CYAN};
                    border-radius: 0px;
                    text-align: left;
                }}
            """)
            self._text_lbl.setStyleSheet(
                f"font-size: 13px; font-weight: 700; background: transparent;"
                f"color: {Colors.CYAN};"
            )
            self._icon_lbl.setStyleSheet(
                f"font-size: 17px; background: transparent; color: {Colors.CYAN};"
            )
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    border: none;
                    border-left: 3px solid transparent;
                    border-radius: 0px;
                    text-align: left;
                }}
                QPushButton:hover {{
                    background-color: {Colors.BG_HOVER};
                }}
            """)
            self._text_lbl.setStyleSheet(
                f"font-size: 13px; font-weight: 500; background: transparent;"
                f"color: {Colors.TEXT_DIM};"
            )
            self._icon_lbl.setStyleSheet(
                f"font-size: 17px; background: transparent; color: {Colors.TEXT_DARK};"
            )


class Sidebar(QWidget):
    page_changed = pyqtSignal(int)

    PAGE_DASHBOARD  = 0
    PAGE_ATTENDANCE = 1
    PAGE_ENROLL     = 2
    PAGE_STUDENTS   = 3
    PAGE_REPORTS    = 4
    PAGE_SETTINGS   = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {Colors.BG_PANEL};
                border-right: 1px solid {Colors.BORDER};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Logo header ──
        header = QWidget()
        header.setFixedHeight(64)
        header.setStyleSheet(f"""
            background-color: {Colors.BG_PANEL};
            border-bottom: 1px solid {Colors.BORDER};
            border-right: none;
        """)
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(16, 0, 16, 0)
        h_lay.setSpacing(10)

        # Logo circle
        logo_wrap = QWidget()
        logo_wrap.setFixedSize(36, 36)
        logo_wrap.setStyleSheet(f"""
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                stop:0 {Colors.CYAN}, stop:1 {Colors.CYAN_DIM});
            border-radius: 10px;
        """)
        logo_inner = QHBoxLayout(logo_wrap)
        logo_inner.setContentsMargins(0, 0, 0, 0)
        logo_lbl = QLabel("👁")
        logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_lbl.setStyleSheet("font-size: 18px; background: transparent;")
        logo_inner.addWidget(logo_lbl)

        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        t1 = QLabel("FaceAttend")
        t1.setStyleSheet(
            f"font-size: 14px; font-weight: 800; color: {Colors.TEXT};"
            f"background: transparent; border: none; letter-spacing: 0.3px;"
        )
        t2 = QLabel("v1.0.0")
        t2.setStyleSheet(
            f"font-size: 10px; color: {Colors.TEXT_DARK};"
            f"background: transparent; border: none;"
        )
        title_col.addWidget(t1)
        title_col.addWidget(t2)

        h_lay.addWidget(logo_wrap)
        h_lay.addLayout(title_col)
        layout.addWidget(header)

        # ── Section label ──
        menu_label = QLabel("ĐIỀU HƯỚNG")
        menu_label.setStyleSheet(f"""
            color: {Colors.TEXT_DARK};
            font-size: 9px;
            font-weight: 700;
            letter-spacing: 2px;
            padding: 14px 20px 4px 20px;
            background: transparent;
            border: none;
        """)
        layout.addWidget(menu_label)

        # ── Nav items ──
        self._nav_buttons: list[NavButton] = []
        nav_items = [
            ("📊", "Dashboard",      self.PAGE_DASHBOARD),
            ("📷", "Điểm Danh",      self.PAGE_ATTENDANCE),
            ("✏️", "Đăng Ký HV",     self.PAGE_ENROLL),
            ("👥", "Học Viên",        self.PAGE_STUDENTS),
            ("📋", "Báo Cáo",        self.PAGE_REPORTS),
        ]
        for icon, label, page_id in nav_items:
            btn = NavButton(icon, label, page_id)
            btn.clicked.connect(lambda checked, pid=page_id: self._on_nav_click(pid))
            self._nav_buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()

        # ── Divider ──
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(
            f"background-color: {Colors.BORDER}; border: none; max-height: 1px;"
        )
        layout.addWidget(sep)

        # ── Settings ──
        settings_btn = NavButton("⚙️", "Cài Đặt", self.PAGE_SETTINGS)
        settings_btn.clicked.connect(lambda: self._on_nav_click(self.PAGE_SETTINGS))
        self._nav_buttons.append(settings_btn)
        layout.addWidget(settings_btn)

        # ── DB Status chip ──
        self._db_chip = QLabel("⬤  Đang kết nối...")
        self._db_chip.setStyleSheet(f"""
            color: {Colors.ORANGE};
            font-size: 11px;
            font-weight: 600;
            padding: 9px 18px 12px 18px;
            background: transparent;
            border: none;
        """)
        layout.addWidget(self._db_chip)

        self._set_active(self.PAGE_DASHBOARD)

    def _on_nav_click(self, page_id: int):
        self._set_active(page_id)
        self.page_changed.emit(page_id)

    def _set_active(self, page_id: int):
        for btn in self._nav_buttons:
            btn.set_active(btn.page_id == page_id)

    def set_db_status(self, connected: bool):
        if connected:
            self._db_chip.setText("⬤  Đã kết nối DB")
            self._db_chip.setStyleSheet(f"""
                color: {Colors.GREEN_DIM};
                font-size: 11px; font-weight: 600;
                padding: 9px 18px 12px 18px;
                background: transparent; border: none;
            """)
        else:
            self._db_chip.setText("⬤  Mất kết nối DB")
            self._db_chip.setStyleSheet(f"""
                color: {Colors.RED};
                font-size: 11px; font-weight: 600;
                padding: 9px 18px 12px 18px;
                background: transparent; border: none;
            """)