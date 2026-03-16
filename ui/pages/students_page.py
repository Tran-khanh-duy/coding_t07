"""
ui/pages/students_page.py
Màn hình danh sách học viên — tìm kiếm, xem, xoá.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QMessageBox, QScrollArea, QFrame,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ui.styles.theme import Colors, card_style, badge_style


class StudentsPage(QWidget):

    # Signal chuyển sang tab Đăng ký — mang student_id (int) hoặc -1 nếu đăng ký mới
    go_to_enroll = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_students = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        # ── Header ──
        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(3)
        title = QLabel("Danh Sách Học Viên")
        title.setStyleSheet(f"font-size: 22px; font-weight: 900; color: {Colors.TEXT};")
        self._subtitle = QLabel("Đang tải...")
        self._subtitle.setStyleSheet(f"font-size: 13px; color: {Colors.TEXT_DIM};")
        title_col.addWidget(title)
        title_col.addWidget(self._subtitle)
        header.addLayout(title_col)
        header.addStretch()

        btn_add = QPushButton("➕  Thêm học viên")
        btn_add.setFixedHeight(38)
        btn_add.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.CYAN};
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 0 16px;
                font-weight: 700;
                font-size: 13px;
            }}
            QPushButton:hover {{ background: {Colors.CYAN_DIM}; }}
        """)
        btn_add.clicked.connect(lambda: self.go_to_enroll.emit(-1))
        header.addWidget(btn_add)

        btn_refresh = QPushButton("🔄")
        btn_refresh.setFixedSize(38, 38)
        btn_refresh.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_CARD};
                color: {Colors.TEXT_DIM};
                border: 1px solid {Colors.BORDER_LT};
                border-radius: 8px;
                font-size: 16px;
            }}
            QPushButton:hover {{ background: {Colors.BG_HOVER}; }}
        """)
        btn_refresh.clicked.connect(self.load_students)
        header.addWidget(btn_refresh)
        layout.addLayout(header)

        # ── Search + Filter bar ──
        bar = QHBoxLayout()
        bar.setSpacing(10)

        self._inp_search = QLineEdit()
        self._inp_search.setPlaceholderText("🔍  Tìm theo mã HV, tên, lớp...")
        self._inp_search.setFixedHeight(38)
        self._inp_search.setStyleSheet(f"""
            QLineEdit {{
                background: {Colors.BG_INPUT};
                color: {Colors.TEXT};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 0 12px;
                font-size: 13px;
            }}
            QLineEdit:focus {{ border-color: {Colors.CYAN}; }}
        """)
        self._inp_search.textChanged.connect(self._filter_table)
        bar.addWidget(self._inp_search, 2)

        self._cmb_filter = QComboBox()
        self._cmb_filter.addItems(["Tất cả", "Đã đăng ký mặt", "Chưa đăng ký"])
        self._cmb_filter.setFixedHeight(38)
        self._cmb_filter.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.BG_INPUT};
                color: {Colors.TEXT};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 0 12px;
                font-size: 13px;
            }}
            QComboBox QAbstractItemView {{
                background: {Colors.BG_CARD};
                color: {Colors.TEXT};
                selection-background-color: {Colors.BG_HOVER};
            }}
        """)
        self._cmb_filter.currentIndexChanged.connect(self._filter_table)
        bar.addWidget(self._cmb_filter, 1)
        layout.addLayout(bar)

        # ── Bảng học viên ──
        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels([
            "Mã HV", "Họ và Tên", "Lớp", "Giới tính",
            "Khuôn mặt", "Ngày tạo", "Thao tác"
        ])
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.setStyleSheet(f"""
            QTableWidget {{
                background: {Colors.BG_PANEL};
                border: 1px solid {Colors.BORDER};
                border-radius: 10px;
                gridline-color: {Colors.BORDER};
                color: {Colors.TEXT};
                font-size: 13px;
                alternate-background-color: {Colors.BG_CARD};
            }}
            QTableWidget::item {{
                padding: 10px 12px;
                border-bottom: 1px solid {Colors.BORDER};
            }}
            QTableWidget::item:selected {{
                background: {Colors.BG_SELECTED};
                color: {Colors.CYAN};
            }}
            QHeaderView::section {{
                background: {Colors.BG_CARD};
                color: {Colors.TEXT_DIM};
                font-weight: 700;
                font-size: 11px;
                letter-spacing: 1px;
                padding: 10px 12px;
                border: none;
                border-bottom: 2px solid {Colors.BORDER_LT};
                text-transform: uppercase;
            }}
        """)
        self._table.setRowHeight(0, 48)
        layout.addWidget(self._table, 1)

        # ── Footer stats ──
        self._footer = QLabel("")
        self._footer.setStyleSheet(f"font-size: 12px; color: {Colors.TEXT_DIM};")
        layout.addWidget(self._footer)

        # Load data sau 200ms
        QTimer.singleShot(200, self.load_students)

    def load_students(self):
        """Load danh sách học viên từ DB."""
        try:
            from database.repositories import student_repo
            self._all_students = student_repo.get_all()
            self._render_table(self._all_students)

            total    = len(self._all_students)
            enrolled = sum(1 for s in self._all_students if s.face_enrolled)
            self._subtitle.setText(
                f"Tổng: {total} học viên  •  Đã đăng ký mặt: {enrolled}  •  Chưa đăng ký: {total-enrolled}"
            )
        except Exception as e:
            logger.error(f"Load students error: {e}")
            self._subtitle.setText(f"Lỗi tải dữ liệu: {e}")

    def _filter_table(self):
        kw     = self._inp_search.text().strip().lower()
        filter_idx = self._cmb_filter.currentIndex()

        filtered = []
        for s in self._all_students:
            # Filter enrolled
            if filter_idx == 1 and not s.face_enrolled:
                continue
            if filter_idx == 2 and s.face_enrolled:
                continue
            # Search
            if kw:
                haystack = f"{s.student_code} {s.full_name} {s.class_name or ''}".lower()
                if kw not in haystack:
                    continue
            filtered.append(s)

        self._render_table(filtered)

    def _render_table(self, students: list):
        self._table.setRowCount(0)
        for row, s in enumerate(students):
            self._table.insertRow(row)
            self._table.setRowHeight(row, 46)

            def cell(text: str, align=Qt.AlignmentFlag.AlignLeft) -> QTableWidgetItem:
                item = QTableWidgetItem(str(text))
                item.setTextAlignment(align | Qt.AlignmentFlag.AlignVCenter)
                return item

            self._table.setItem(row, 0, cell(s.student_code,
                Qt.AlignmentFlag.AlignCenter))
            self._table.setItem(row, 1, cell(s.full_name))
            self._table.setItem(row, 2, cell(s.class_name or "—"))
            self._table.setItem(row, 3, cell(s.gender or "—",
                Qt.AlignmentFlag.AlignCenter))

            # Khuôn mặt badge
            enrolled = s.face_enrolled
            badge = QTableWidgetItem("✅ Đã đăng ký" if enrolled else "⬜ Chưa đăng ký")
            badge.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
            badge.setForeground(QColor(Colors.GREEN if enrolled else Colors.TEXT_DARK))
            self._table.setItem(row, 4, badge)

            created = getattr(s, 'created_at', None)
            date_str = created.strftime("%d/%m/%Y") if created else "—"
            self._table.setItem(row, 5, cell(date_str, Qt.AlignmentFlag.AlignCenter))

            # Nút thao tác
            action_widget = QWidget()
            action_layout = QHBoxLayout(action_widget)
            action_layout.setContentsMargins(6, 4, 6, 4)
            action_layout.setSpacing(6)

            btn_enroll = QPushButton("📷 Đăng ký mặt" if not enrolled else "🔄 Cập nhật")
            btn_enroll.setFixedHeight(30)
            btn_enroll.setStyleSheet(f"""
                QPushButton {{
                    background: {Colors.GREEN_DIM if enrolled else Colors.CYAN};
                    color: #ffffff;
                    border: none;
                    border-radius: 6px;
                    padding: 0 10px;
                    font-size: 11px;
                    font-weight: 700;
                }}
                QPushButton:hover {{ background: {Colors.GREEN if enrolled else Colors.CYAN_DIM}; }}
            """)
            btn_enroll.clicked.connect(
                lambda _, sid=s.student_id: self.go_to_enroll.emit(sid)
            )
            action_layout.addWidget(btn_enroll)
            action_layout.addStretch()
            self._table.setCellWidget(row, 6, action_widget)

        count = len(students)
        total = len(self._all_students)
        self._footer.setText(
            f"Hiển thị {count} / {total} học viên"
        )