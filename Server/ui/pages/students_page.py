"""
ui/pages/students_page.py
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QLineEdit, QTableWidget, QTableWidgetItem, 
    QHeaderView, QComboBox, QMessageBox, QFrame,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.styles.theme import Colors, card_style, badge_style, combo_style, input_style


class StudentsPage(QWidget):
    # Signal chuyển sang tab Đăng ký — mang student_id (int) hoặc -1 nếu đăng ký mới
    go_to_enroll = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_students = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(20)

        # ── Header ──
        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        
        title = QLabel("Danh Sách Học Viên")
        title.setStyleSheet(f"font-size: 26px; font-weight: 800; color: {Colors.TEXT};")
        
        self._subtitle = QLabel("Đang tải dữ liệu học viên...")
        self._subtitle.setStyleSheet(f"font-size: 14px; color: {Colors.TEXT_DIM};")
        
        title_col.addWidget(title)
        title_col.addWidget(self._subtitle)
        header.addLayout(title_col)
        header.addStretch()

        # Nút Thêm mới
        btn_add = QPushButton("➕  THÊM HỌC VIÊN MỚI")
        btn_add.setFixedHeight(45)
        btn_add.setMinimumWidth(200)
        btn_add.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.CYAN}; color: white;
                border: none; border-radius: 10px;
                font-weight: 800; font-size: 13px; letter-spacing: 0.5px;
            }}
            QPushButton:hover {{ background: {Colors.CYAN_DIM}; }}
        """)
        btn_add.clicked.connect(lambda: self.go_to_enroll.emit(-1))
        header.addWidget(btn_add)

        # Nút Làm mới
        btn_refresh = QPushButton("🔄")
        btn_refresh.setFixedSize(45, 45)
        btn_refresh.setToolTip("Làm mới danh sách")
        btn_refresh.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_CARD}; color: {Colors.TEXT};
                border: 1px solid {Colors.BORDER_LT}; border-radius: 10px;
                font-size: 18px;
            }}
            QPushButton:hover {{ background: {Colors.BG_HOVER}; border-color: {Colors.CYAN}; }}
        """)
        btn_refresh.clicked.connect(self.load_students)
        header.addWidget(btn_refresh)
        
        layout.addLayout(header)

        # ── Search + Filter Bar ──
        search_card = QFrame()
        search_card.setStyleSheet(card_style(Colors.BORDER, radius=12))
        search_lay = QHBoxLayout(search_card)
        search_lay.setContentsMargins(15, 10, 15, 10)
        search_lay.setSpacing(15)

        # Ô tìm kiếm
        self._inp_search = QLineEdit()
        self._inp_search.setPlaceholderText("🔍  Tìm kiếm mã học viên, tên hoặc lớp...")
        self._inp_search.setStyleSheet(input_style())
        self._inp_search.textChanged.connect(self._filter_table)
        search_lay.addWidget(self._inp_search, 3)

        # Bộ lọc trạng thái
        self._cmb_filter = QComboBox()
        self._cmb_filter.addItems(["💎 Tất cả học viên", "✅ Đã đăng ký khuôn mặt", "⚠️ Chưa đăng ký mặt"])
        self._cmb_filter.setStyleSheet(combo_style())
        self._cmb_filter.currentIndexChanged.connect(self._filter_table)
        search_lay.addWidget(self._cmb_filter, 1)
        
        layout.addWidget(search_card)

        # ── Bảng dữ liệu ──
        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels([
            "MÃ HỌC VIÊN", "HỌ VÀ TÊN", "LỚP HỌC", "GIỚI TÍNH",
            "TRẠNG THÁI", "NGÀY TẠO", "THAO TÁC"
        ])
        
        # Cấu hình Header
        header_view = self._table.horizontalHeader()
        header_view.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents) # Mã HV
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)          # Tên HV
        header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents) # Lớp học
        header_view.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents) # Giới tính
        
        # Trạng thái và Thao tác cố định độ rộng để không bị cắt chữ widget
        header_view.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(4, 160)
        
        header_view.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents) # Ngày tạo
        
        header_view.setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(6, 260) # Cột thao tác rộng rãi

        
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(78)
        self._table.verticalHeader().setMinimumSectionSize(78)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(False)
        self._table.setShowGrid(False)
        
        # Style bảng đồng bộ với Theme
        self._table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {Colors.BG_PANEL};
                border: 1px solid {Colors.BORDER};
                border-radius: 12px;
                color: {Colors.TEXT};
                font-size: 14px;
            }}
            QTableWidget::item {{
                padding: 0px 12px;
                border-bottom: 1px solid {Colors.BG_DARK};
            }}
            QTableWidget::item:selected {{
                background-color: {Colors.BG_SELECTED};
                color: {Colors.CYAN};
                font-weight: 600;
            }}
            QHeaderView::section {{
                background-color: {Colors.BG_CARD};
                color: {Colors.TEXT_DIM};
                font-weight: 800;
                font-size: 11px;
                padding: 12px;
                border: none;
                border-bottom: 2px solid {Colors.BORDER_LT};
                text-transform: uppercase;
            }}
        """)
        self._table.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self._table, 1)

        # ── Footer ──
        footer_lay = QHBoxLayout()
        self._lbl_footer = QLabel("Hiển thị 0 học viên")
        self._lbl_footer.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {Colors.TEXT_DIM};")
        footer_lay.addWidget(self._lbl_footer)
        footer_lay.addStretch()
        
        lbl_hint = QLabel("💡 Mẹo: Nhấn đúp vào hàng để xem chi tiết")
        lbl_hint.setStyleSheet(f"font-size: 12px; color: {Colors.TEXT_DARK}; font-style: italic;")
        footer_lay.addWidget(lbl_hint)
        
        layout.addLayout(footer_lay)

        # Tự động tải dữ liệu
        QTimer.singleShot(100, self.load_students)

    def load_students(self):
        """Tải dữ liệu từ Repository."""
        try:
            from database.repositories import student_repo
            self._all_students = student_repo.get_all()
            self._render_table(self._all_students)

            total = len(self._all_students)
            enrolled = sum(1 for s in self._all_students if s.face_enrolled)
            self._subtitle.setText(
                f"Tổng cộng: {total} học viên  |  Đã đăng ký: {enrolled}  |  Chưa đăng ký: {total-enrolled}"
            )
        except Exception as e:
            logger.error(f"Lỗi tải danh sách học viên: {e}")
            self._subtitle.setText(f"❌ Không thể kết nối cơ sở dữ liệu")

    def _filter_table(self):
        """Bộ lọc tìm kiếm và trạng thái."""
        keyword = self._inp_search.text().strip().lower()
        filter_idx = self._cmb_filter.currentIndex()

        filtered = []
        for s in self._all_students:
            # 1. Lọc theo trạng thái mặt
            if filter_idx == 1 and not s.face_enrolled: continue
            if filter_idx == 2 and s.face_enrolled: continue
            
            # 2. Lọc theo từ khóa
            if keyword:
                search_pool = f"{s.student_code} {s.full_name} {s.class_name or ''}".lower()
                if keyword not in search_pool: continue
            
            filtered.append(s)

        self._render_table(filtered)

    def _render_table(self, students: list):
        self._table.setRowCount(0)
        self._displayed_students = students # Lưu lại để tham chiếu ID khi click
        self._lbl_footer.setText(f"Hiển thị {len(students)} / {len(self._all_students)} học viên")

        for row, s in enumerate(students):
            self._table.insertRow(row)
            self._table.setRowHeight(row, 75) # Tăng độ cao hàng để cell widget không bị cắt mấp mé

            # 1. Mã HV (Căn giữa, Font đậm)
            item_code = QTableWidgetItem(s.student_code)
            item_code.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_code.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            self._table.setItem(row, 0, item_code)

            # 2. Họ Tên
            self._table.setItem(row, 1, QTableWidgetItem(s.full_name))

            # 3. Lớp
            item_class = QTableWidgetItem(s.class_name or "Chưa xếp lớp")
            item_class.setForeground(QColor(Colors.TEXT_DIM if not s.class_name else Colors.TEXT))
            self._table.setItem(row, 2, item_class)

            # 4. Giới tính
            item_gender = QTableWidgetItem(s.gender or "—")
            item_gender.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 3, item_gender)

            # 5. Badge Trạng thái khuôn mặt
            status_widget = QWidget()
            status_widget.setStyleSheet("background: transparent;")
            status_lay = QHBoxLayout(status_widget)
            status_lay.setContentsMargins(0, 0, 0, 0)
            status_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            enrolled = s.face_enrolled
            badge = QLabel("ĐÃ ĐĂNG KÝ" if enrolled else "CHƯA CÓ MẶT")
            badge.setStyleSheet(badge_style(Colors.GREEN if enrolled else Colors.TEXT_DARK))
            badge.setMinimumWidth(125)
            badge.setMinimumHeight(32)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status_lay.addWidget(badge)
            self._table.setCellWidget(row, 4, status_widget)

            # 6. Ngày tạo
            date_val = s.created_at.strftime("%d/%m/%Y") if hasattr(s, 'created_at') and s.created_at else "—"
            item_date = QTableWidgetItem(date_val)
            item_date.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 5, item_date)

            # 7. Nhóm nút Thao tác
            action_widget = QWidget()
            action_widget.setStyleSheet("background: transparent;")
            action_lay = QHBoxLayout(action_widget)
            action_lay.setContentsMargins(10, 4, 10, 4) # Giảm margin dọc
            action_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
            action_lay.setSpacing(8)

            # Nút Đăng ký/Cập nhật
            btn_edit = QPushButton("📷 ĐĂNG KÝ" if not enrolled else "🔄 CẬP NHẬT")
            btn_edit.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_edit.setMinimumHeight(34)
            btn_edit.setStyleSheet(f"""
                QPushButton {{
                    background: {Colors.CYAN if not enrolled else Colors.BG_HOVER};
                    color: {"white" if not enrolled else Colors.CYAN};
                    border: 1px solid {Colors.CYAN}; border-radius: 6px;
                    font-size: 11px; font-weight: 800; padding: 5px 12px;
                }}
                QPushButton:hover {{ background: {Colors.CYAN_DIM}; color: white; }}
            """)
            btn_edit.clicked.connect(lambda _, sid=s.student_id: self.go_to_enroll.emit(sid))
            
            # Nút Xóa
            btn_del = QPushButton("🗑️")
            btn_del.setToolTip("Xóa học viên")
            btn_del.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_del.setMinimumHeight(34)
            btn_del.setMinimumWidth(36)
            btn_del.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; color: {Colors.RED};
                    border: 1px solid {Colors.RED}44; border-radius: 6px;
                    font-size: 15px; padding: 4px;
                }}
                QPushButton:hover {{ background: {Colors.RED}; color: white; }}
            """)
            btn_del.clicked.connect(lambda _, sid=s.student_id, name=s.full_name: self._on_delete_student(sid, name))

            action_lay.addWidget(btn_edit)
            action_lay.addWidget(btn_del)
            self._table.setCellWidget(row, 6, action_widget)

    def _on_item_double_clicked(self, item):
        """Xử lý double click vào hàng để mở trang đăng ký/cập nhật."""
        row = item.row()
        if hasattr(self, '_displayed_students') and row < len(self._displayed_students):
            student = self._displayed_students[row]
            self.go_to_enroll.emit(student.student_id)

    def _on_delete_student(self, student_id: int, name: str):
        """Xử lý xóa học viên."""
        confirm = QMessageBox.question(
            self, "Xác nhận xóa",
            f"Bạn có chắc chắn muốn xóa học viên <b>{name}</b>?<br>"
            "Dữ liệu khuôn mặt và lịch sử điểm danh liên quan cũng sẽ bị xóa.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if confirm == QMessageBox.StandardButton.Yes:
            try:
                from database.repositories import student_repo
                from services.embedding_cache_manager import cache_manager
                if student_repo.delete(student_id):
                    logger.success(f"Đã xóa học viên ID: {student_id}")
                    # Xóa khỏi RAM cache ngay lập tức
                    cache_manager.remove_student_from_cache(student_id)
                    self.load_students() # Tải lại bảng
                else:
                    QMessageBox.warning(self, "Lỗi", "Không thể xóa học viên này!")
            except Exception as e:
                logger.error(f"Delete student error: {e}")
                QMessageBox.critical(self, "Lỗi hệ thống", str(e))

    def showEvent(self, event):
        """Mỗi khi tab được hiển thị, tự động làm mới dữ liệu."""
        self.load_students()
        super().showEvent(event)