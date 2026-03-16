"""
ui/pages/reports_page.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Màn hình Báo Cáo

Layout:
  ┌──────────────────────┬──────────────────────────┐
  │  Danh sách buổi học  │  Chi tiết buổi đã chọn   │
  │  (filter ngày/lớp)   │  - Thông tin tổng quan    │
  │                      │  - Bảng điểm danh preview │
  │                      │  [Xuất Excel] [Xuất PDF]  │
  └──────────────────────┴──────────────────────────┘
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os
import subprocess
from datetime import datetime, timedelta

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QScrollArea, QFrame, QProgressBar,
    QMessageBox, QFileDialog, QDateEdit,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QDate
from PyQt6.QtGui import QColor, QFont
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ui.styles.theme import Colors, card_style


# ─────────────────────────────────────────────
#  Export Worker
# ─────────────────────────────────────────────
class ExportWorker(QThread):
    done    = pyqtSignal(dict)
    progress= pyqtSignal(str)

    def __init__(self, session_id: int, fmt: str, parent=None):
        super().__init__(parent)
        self.session_id = session_id
        self.fmt        = fmt

    def run(self):
        try:
            self.progress.emit(f"Đang tải dữ liệu session {self.session_id}...")
            from services.report_service import generate_report
            self.progress.emit("Đang tạo file...")
            result = generate_report(self.session_id, self.fmt)
            self.done.emit(result)
        except Exception as e:
            self.done.emit({"success": False, "error": str(e)})


# ─────────────────────────────────────────────
#  StatCard nhỏ cho preview
# ─────────────────────────────────────────────
class MiniStat(QWidget):
    def __init__(self, label: str, value: str, color: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            background: {color}15;
            border: 1px solid {color}40;
            border-radius: 8px;
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)

        self._val = QLabel(value)
        self._val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._val.setStyleSheet(f"font-size: 22px; font-weight: 900; color: {color};")

        lbl = QLabel(label)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(f"font-size: 10px; color: {Colors.TEXT_DIM};")

        lay.addWidget(self._val)
        lay.addWidget(lbl)

    def set_value(self, v: str):
        self._val.setText(v)


# ─────────────────────────────────────────────
#  ReportsPage
# ─────────────────────────────────────────────
class ReportsPage(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._selected_session_id: int | None = None
        self._export_worker: ExportWorker | None = None
        self._setup_ui()

    # ─── UI ───────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        # ── Header ──
        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(3)
        title = QLabel("Báo Cáo Điểm Danh")
        title.setStyleSheet(f"font-size: 22px; font-weight: 900; color: {Colors.TEXT};")
        sub = QLabel("Chọn buổi học để xem chi tiết và xuất báo cáo")
        sub.setStyleSheet(f"font-size: 13px; color: {Colors.TEXT_DIM};")
        title_col.addWidget(title)
        title_col.addWidget(sub)
        header.addLayout(title_col)
        header.addStretch()

        btn_refresh = QPushButton("🔄  Tải lại")
        btn_refresh.setFixedHeight(36)
        btn_refresh.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_CARD};
                color: {Colors.TEXT_DIM};
                border: 1px solid {Colors.BORDER_LT};
                border-radius: 8px;
                padding: 0 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{ color: {Colors.TEXT}; background: {Colors.BG_HOVER}; }}
        """)
        btn_refresh.clicked.connect(self._load_sessions)
        header.addWidget(btn_refresh)
        root.addLayout(header)

        # ── Filter bar ──
        filter_bar = QHBoxLayout()
        filter_bar.setSpacing(10)

        self._inp_search = QLineEdit()
        self._inp_search.setPlaceholderText("🔍  Tìm môn học, lớp...")
        self._inp_search.setFixedHeight(36)
        self._inp_search.setStyleSheet(self._input_style())
        self._inp_search.textChanged.connect(self._filter_sessions)

        self._cmb_class = QComboBox()
        self._cmb_class.setFixedHeight(36)
        self._cmb_class.setFixedWidth(200)
        self._cmb_class.setStyleSheet(self._combo_style())
        self._cmb_class.currentIndexChanged.connect(self._filter_sessions)

        self._cmb_status = QComboBox()
        self._cmb_status.addItems(["Tất cả trạng thái", "COMPLETED", "ACTIVE", "PENDING"])
        self._cmb_status.setFixedHeight(36)
        self._cmb_status.setFixedWidth(160)
        self._cmb_status.setStyleSheet(self._combo_style())
        self._cmb_status.currentIndexChanged.connect(self._filter_sessions)

        filter_bar.addWidget(self._inp_search, 2)
        filter_bar.addWidget(self._cmb_class)
        filter_bar.addWidget(self._cmb_status)
        root.addLayout(filter_bar)

        # ── Splitter: Left list / Right detail ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background: transparent; width: 10px; }")

        left  = self._build_session_list()
        right = self._build_detail_panel()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([460, 640])
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        # Load data
        QTimer.singleShot(300, self._load_sessions)

    def _build_session_list(self) -> QWidget:
        panel = QWidget()
        lay   = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        lbl = QLabel("DANH SÁCH BUỔI HỌC")
        lbl.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {Colors.TEXT_DIM}; letter-spacing: 1.5px;"
        )
        lay.addWidget(lbl)

        self._session_table = QTableWidget()
        self._session_table.setColumnCount(5)
        self._session_table.setHorizontalHeaderLabels([
            "Ngày", "Lớp", "Môn học", "Tỉ lệ", "Trạng thái"
        ])
        hh = self._session_table.horizontalHeader()
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._session_table.verticalHeader().setVisible(False)
        self._session_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._session_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._session_table.setShowGrid(False)
        self._session_table.setStyleSheet(f"""
            QTableWidget {{
                background: {Colors.BG_PANEL};
                border: 1px solid {Colors.BORDER};
                border-radius: 10px;
                color: {Colors.TEXT};
                font-size: 12px;
                alternate-background-color: {Colors.BG_CARD};
            }}
            QTableWidget::item {{
                padding: 8px 10px;
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
                font-size: 10px;
                letter-spacing: 1px;
                padding: 8px 10px;
                border: none;
                border-bottom: 2px solid {Colors.BORDER_LT};
            }}
        """)
        self._session_table.setAlternatingRowColors(True)
        self._session_table.itemSelectionChanged.connect(self._on_session_selected)
        lay.addWidget(self._session_table, 1)

        self._session_count_lbl = QLabel("")
        self._session_count_lbl.setStyleSheet(f"font-size: 11px; color: {Colors.TEXT_DIM};")
        lay.addWidget(self._session_count_lbl)
        return panel

    def _build_detail_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(card_style(Colors.BORDER))
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(14)

        # Placeholder khi chưa chọn
        self._placeholder = QWidget()
        ph_lay = QVBoxLayout(self._placeholder)
        ph_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph_lay.setSpacing(10)
        ph_icon = QLabel("📋")
        ph_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph_icon.setStyleSheet("font-size: 48px;")
        ph_txt = QLabel("Chọn một buổi học\nở danh sách bên trái")
        ph_txt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph_txt.setStyleSheet(f"font-size: 14px; color: {Colors.TEXT_DIM};")
        ph_lay.addWidget(ph_icon)
        ph_lay.addWidget(ph_txt)
        lay.addWidget(self._placeholder, 1)

        # Panel chi tiết (ẩn khi chưa chọn)
        self._detail = QWidget()
        self._detail.hide()
        detail_lay = QVBoxLayout(self._detail)
        detail_lay.setContentsMargins(0, 0, 0, 0)
        detail_lay.setSpacing(14)

        # Tiêu đề buổi
        self._lbl_session_title = QLabel()
        self._lbl_session_title.setStyleSheet(
            f"font-size: 16px; font-weight: 900; color: {Colors.TEXT};"
        )
        self._lbl_session_title.setWordWrap(True)
        detail_lay.addWidget(self._lbl_session_title)

        # Info row
        self._lbl_info = QLabel()
        self._lbl_info.setStyleSheet(f"font-size: 12px; color: {Colors.TEXT_DIM};")
        detail_lay.addWidget(self._lbl_info)

        # Stats
        stats_row = QHBoxLayout()
        stats_row.setSpacing(10)
        self._stat_total   = MiniStat("Tổng HV",  "—", Colors.CYAN)
        self._stat_present = MiniStat("Có mặt",   "—", Colors.GREEN)
        self._stat_absent  = MiniStat("Vắng mặt", "—", Colors.RED)
        self._stat_rate    = MiniStat("Tỉ lệ",    "—%", Colors.ORANGE)
        for w in [self._stat_total, self._stat_present, self._stat_absent, self._stat_rate]:
            stats_row.addWidget(w)
        detail_lay.addLayout(stats_row)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"background: {Colors.BORDER}; max-height: 1px;")
        detail_lay.addWidget(line)

        # Preview bảng điểm danh
        preview_hdr = QLabel("XEM TRƯỚC ĐIỂM DANH")
        preview_hdr.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {Colors.TEXT_DIM}; letter-spacing: 1.5px;"
        )
        detail_lay.addWidget(preview_hdr)

        self._preview_table = QTableWidget()
        self._preview_table.setColumnCount(5)
        self._preview_table.setHorizontalHeaderLabels([
            "Mã HV", "Họ và Tên", "Trạng thái", "Giờ vào", "Chính xác"
        ])
        ph2 = self._preview_table.horizontalHeader()
        ph2.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._preview_table.verticalHeader().setVisible(False)
        self._preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._preview_table.setShowGrid(False)
        self._preview_table.setMaximumHeight(260)
        self._preview_table.setStyleSheet(f"""
            QTableWidget {{
                background: {Colors.BG_PANEL};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                color: {Colors.TEXT};
                font-size: 12px;
                alternate-background-color: {Colors.BG_CARD};
            }}
            QTableWidget::item {{
                padding: 6px 8px;
                border-bottom: 1px solid {Colors.BORDER};
            }}
            QHeaderView::section {{
                background: {Colors.BG_CARD};
                color: {Colors.TEXT_DIM};
                font-weight: 700;
                font-size: 10px;
                padding: 7px 8px;
                border: none;
                border-bottom: 1px solid {Colors.BORDER_LT};
            }}
        """)
        self._preview_table.setAlternatingRowColors(True)
        detail_lay.addWidget(self._preview_table, 1)

        # Progress bar khi export
        self._export_progress = QLabel("")
        self._export_progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._export_progress.setStyleSheet(
            f"font-size: 12px; color: {Colors.CYAN}; font-weight: 600;"
        )
        self._export_progress.hide()
        detail_lay.addWidget(self._export_progress)

        # Nút xuất
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self._btn_excel = QPushButton("📊  Xuất Excel (.xlsx)")
        self._btn_excel.setFixedHeight(44)
        self._btn_excel.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.GREEN};
                color: #ffffff;
                border: none;
                border-radius: 9px;
                font-size: 13px;
                font-weight: 700;
            }}
            QPushButton:hover {{ background: {Colors.GREEN_DIM}; }}
            QPushButton:disabled {{
                background: {Colors.BORDER};
                color: {Colors.TEXT_DARK};
            }}
        """)
        self._btn_excel.clicked.connect(lambda: self._export("excel"))

        self._btn_pdf = QPushButton("📄  Xuất PDF")
        self._btn_pdf.setFixedHeight(44)
        self._btn_pdf.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.RED};
                color: #ffffff;
                border: none;
                border-radius: 9px;
                font-size: 13px;
                font-weight: 700;
            }}
            QPushButton:hover {{ background: {Colors.RED_DIM}; }}
            QPushButton:disabled {{
                background: {Colors.BORDER};
                color: {Colors.TEXT_DARK};
            }}
        """)
        self._btn_pdf.clicked.connect(lambda: self._export("pdf"))

        self._btn_both = QPushButton("⬇️  Xuất cả 2")
        self._btn_both.setFixedHeight(44)
        self._btn_both.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.CYAN};
                color: #ffffff;
                border: none;
                border-radius: 9px;
                font-size: 13px;
                font-weight: 700;
            }}
            QPushButton:hover {{ background: {Colors.CYAN_DIM}; }}
            QPushButton:disabled {{
                background: {Colors.BORDER};
                color: {Colors.TEXT_DARK};
            }}
        """)
        self._btn_both.clicked.connect(lambda: self._export("both"))

        btn_row.addWidget(self._btn_excel)
        btn_row.addWidget(self._btn_pdf)
        btn_row.addWidget(self._btn_both)
        detail_lay.addLayout(btn_row)

        # Nút mở thư mục
        self._btn_open_folder = QPushButton("📂  Mở thư mục báo cáo")
        self._btn_open_folder.setFixedHeight(34)
        self._btn_open_folder.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {Colors.TEXT_DIM};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{ color: {Colors.CYAN}; border-color: {Colors.CYAN_DIM}; }}
        """)
        self._btn_open_folder.clicked.connect(self._open_reports_folder)
        detail_lay.addWidget(self._btn_open_folder)

        lay.addWidget(self._detail, 1)
        return panel

    # ─── Logic ────────────────────────────────

    def _load_sessions(self):
        """Load danh sách buổi học từ DB."""
        self._all_sessions = []
        self._cmb_class.blockSignals(True)
        self._cmb_class.clear()
        self._cmb_class.addItem("Tất cả lớp", None)

        try:
            from database.repositories import session_repo, class_repo
            self._all_sessions = session_repo.get_all()
            classes = class_repo.get_all()
            for cls in classes:
                self._cmb_class.addItem(
                    f"{cls.class_code} — {cls.class_name}", cls.class_id
                )
        except Exception as e:
            logger.error(f"Load sessions error: {e}")

        self._cmb_class.blockSignals(False)
        self._render_sessions(self._all_sessions)

    def _filter_sessions(self):
        kw         = self._inp_search.text().strip().lower()
        class_id   = self._cmb_class.currentData()
        status_txt = self._cmb_status.currentText()

        filtered = []
        for s in self._all_sessions:
            if class_id and getattr(s, "class_id", None) != class_id:
                continue
            if status_txt != "Tất cả trạng thái" and s.status != status_txt:
                continue
            if kw:
                haystack = f"{s.subject_name} {getattr(s,'class_code','')} {getattr(s,'class_name','')}".lower()
                if kw not in haystack:
                    continue
            filtered.append(s)

        self._render_sessions(filtered)

    def _render_sessions(self, sessions: list):
        self._session_table.setRowCount(0)
        for row, s in enumerate(sessions):
            self._session_table.insertRow(row)
            self._session_table.setRowHeight(row, 44)

            def cell(txt, align=Qt.AlignmentFlag.AlignCenter):
                it = QTableWidgetItem(str(txt))
                it.setTextAlignment(align | Qt.AlignmentFlag.AlignVCenter)
                return it

            date_str = s.session_date.strftime("%d/%m/%Y") if s.session_date else "—"
            self._session_table.setItem(row, 0, cell(date_str))
            self._session_table.setItem(row, 1, cell(getattr(s, "class_code", "—")))
            self._session_table.setItem(row, 2, cell(
                s.subject_name, Qt.AlignmentFlag.AlignLeft))

            # Tỉ lệ
            # total_students không có trong schema — tính từ present + absent
            present = getattr(s, "present_count", 0) or 0
            absent  = getattr(s, "absent_count",  0) or 0
            total   = present + absent
            rate    = f"{present/total*100:.0f}%" if total > 0 else "—"
            rate_item = QTableWidgetItem(rate)
            rate_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
            if total > 0:
                pct = present / total
                color = Colors.GREEN if pct >= 0.8 else \
                        Colors.ORANGE if pct >= 0.6 else Colors.RED
                rate_item.setForeground(QColor(color))
            self._session_table.setItem(row, 3, rate_item)

            # Status badge
            status = s.status
            status_colors = {
                "COMPLETED": Colors.GREEN,
                "ACTIVE":    Colors.CYAN,
                "PENDING":   Colors.ORANGE,
            }
            color    = status_colors.get(status, Colors.TEXT_DIM)
            labels   = {"COMPLETED": "✓ Hoàn thành", "ACTIVE": "● Đang chạy", "PENDING": "○ Chờ"}
            st_item  = QTableWidgetItem(labels.get(status, status))
            st_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
            st_item.setForeground(QColor(color))
            self._session_table.setItem(row, 4, st_item)

            # Lưu session_id vào row data
            self._session_table.item(row, 0).setData(Qt.ItemDataRole.UserRole, s.session_id)

        count = len(sessions)
        total = len(self._all_sessions)
        self._session_count_lbl.setText(f"Hiển thị {count} / {total} buổi học")

    def _on_session_selected(self):
        rows = self._session_table.selectedItems()
        if not rows:
            return
        row = self._session_table.currentRow()
        item0 = self._session_table.item(row, 0)
        if not item0:
            return
        sid = item0.data(Qt.ItemDataRole.UserRole)
        if sid:
            self._selected_session_id = sid
            self._load_session_detail(sid)

    def _load_session_detail(self, session_id: int):
        """Load chi tiết và hiển thị preview."""
        try:
            from services.report_service import load_report_data
            data = load_report_data(session_id)
            if not data:
                return

            self._placeholder.hide()
            self._detail.show()

            self._lbl_session_title.setText(
                f"📚  {data.subject_name}"
            )
            self._lbl_info.setText(
                f"🏫  {data.class_code} — {data.class_name}  •  "
                f"📅  {data.session_date}  •  "
                f"⏱  {data.start_time} → {data.end_time}"
            )

            self._stat_total.set_value(str(data.total_students))
            self._stat_present.set_value(str(data.present_count))
            self._stat_absent.set_value(str(data.absent_count))
            self._stat_rate.set_value(f"{data.attendance_rate:.1f}%")

            # Preview table
            records = data.records or []
            self._preview_table.setRowCount(0)
            for row, r in enumerate(records):
                self._preview_table.insertRow(row)
                self._preview_table.setRowHeight(row, 36)

                is_p   = r.get("status") == "PRESENT"
                color  = Colors.GREEN if is_p else Colors.RED

                for col, txt in enumerate([
                    r.get("student_code", ""),
                    r.get("full_name", ""),
                    "✓ Có mặt" if is_p else "✗ Vắng",
                    r.get("check_in_time", "—"),
                    f"{r.get('recognition_score',0)*100:.1f}%" if is_p else "—",
                ]):
                    it = QTableWidgetItem(str(txt))
                    it.setTextAlignment(
                        (Qt.AlignmentFlag.AlignLeft if col == 1
                         else Qt.AlignmentFlag.AlignCenter)
                        | Qt.AlignmentFlag.AlignVCenter
                    )
                    if col == 2:
                        it.setForeground(QColor(color))
                    self._preview_table.setItem(row, col, it)

            self._btn_excel.setEnabled(True)
            self._btn_pdf.setEnabled(True)
            self._btn_both.setEnabled(True)

        except Exception as e:
            logger.error(f"Load detail error: {e}")

    # ─── Export ───────────────────────────────

    def _export(self, fmt: str):
        if not self._selected_session_id:
            QMessageBox.warning(self, "Chưa chọn buổi", "Vui lòng chọn một buổi học!")
            return

        # Disable buttons
        for btn in [self._btn_excel, self._btn_pdf, self._btn_both]:
            btn.setEnabled(False)
        self._export_progress.setText("⏳  Đang xuất file...")
        self._export_progress.show()

        self._export_worker = ExportWorker(self._selected_session_id, fmt)
        self._export_worker.progress.connect(
            lambda msg: self._export_progress.setText(f"⏳  {msg}")
        )
        self._export_worker.done.connect(self._on_export_done)
        self._export_worker.start()

    def _on_export_done(self, result: dict):
        self._export_progress.hide()
        for btn in [self._btn_excel, self._btn_pdf, self._btn_both]:
            btn.setEnabled(True)

        if not result.get("success"):
            QMessageBox.critical(
                self, "Xuất file thất bại",
                f"Lỗi: {result.get('error', 'Không rõ')}"
            )
            return

        # Hiển thị kết quả
        files = []
        if result.get("excel"):
            files.append(f"📊 Excel: {Path(result['excel']).name}")
        if result.get("pdf"):
            files.append(f"📄 PDF:   {Path(result['pdf']).name}")

        msg = QMessageBox(self)
        msg.setWindowTitle("✅ Xuất file thành công!")
        msg.setText(
            "Đã tạo báo cáo thành công!\n\n" +
            "\n".join(files) +
            f"\n\nThư mục: {self._get_reports_dir()}"
        )
        msg.setStandardButtons(
            QMessageBox.StandardButton.Open |
            QMessageBox.StandardButton.Ok
        )
        msg.button(QMessageBox.StandardButton.Open).setText("📂 Mở thư mục")
        ret = msg.exec()
        if ret == QMessageBox.StandardButton.Open:
            self._open_reports_folder()

    def _open_reports_folder(self):
        folder = self._get_reports_dir()
        Path(folder).mkdir(parents=True, exist_ok=True)
        try:
            import subprocess, sys
            if sys.platform == "win32":
                os.startfile(folder)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as e:
            logger.error(f"Open folder error: {e}")

    def _get_reports_dir(self) -> str:
        try:
            from config import app_config
            return str(app_config.reports_dir)
        except Exception:
            return "reports"

    def showEvent(self, event):
        self._load_sessions()
        super().showEvent(event)

    # ─── Style helpers ────────────────────────

    def _input_style(self) -> str:
        return f"""
            QLineEdit {{
                background: {Colors.BG_INPUT};
                color: {Colors.TEXT};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 0 12px;
                font-size: 13px;
            }}
            QLineEdit:focus {{ border-color: {Colors.CYAN}; }}
        """

    def _combo_style(self) -> str:
        return f"""
            QComboBox {{
                background: {Colors.BG_INPUT};
                color: {Colors.TEXT};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 0 12px;
                font-size: 13px;
            }}
            QComboBox:focus {{ border-color: {Colors.CYAN}; }}
            QComboBox QAbstractItemView {{
                background: {Colors.BG_CARD};
                color: {Colors.TEXT};
                selection-background-color: {Colors.BG_HOVER};
            }}
        """