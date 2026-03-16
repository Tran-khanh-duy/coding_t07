"""
ui/pages/attendance_page.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Màn hình Điểm Danh Realtime

Layout:
  ┌──────────────────────────┬────────────────────┐
  │   Live Camera (lớn)      │  Panel điều khiển  │
  │   + bounding box         │  - Chọn lớp/môn    │
  │   + tên + score          │  - [Bắt đầu]       │
  │                          │  - Đồng hồ         │
  │                          ├────────────────────┤
  │                          │  Danh sách có mặt  │
  │                          │  (scroll realtime) │
  │                          │  - Ảnh + tên + giờ │
  └──────────────────────────┴────────────────────┘
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import numpy as np
from datetime import datetime, date

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QFrame,
    QScrollArea, QSizePolicy, QMessageBox,
    QSplitter, QProgressBar, QDateEdit,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QDate
from PyQt6.QtGui import QFont, QColor, QPixmap, QImage
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.styles.theme import Colors, card_style
from ui.widgets.camera_preview import CameraPreviewWidget


# ─────────────────────────────────────────────
#  Worker: chạy AI pipeline trong QThread
# ─────────────────────────────────────────────
class AttendanceWorker(QThread):
    """
    Thread chạy vòng lặp: Camera → FaceEngine → AttendanceService.
    Phát signals lên UI thread an toàn.
    """
    frame_ready     = pyqtSignal(np.ndarray, float)   # (annotated_frame, ms)
    attendance_done = pyqtSignal(dict)                 # event dict
    stats_updated   = pyqtSignal(dict)                 # stats
    error_occurred  = pyqtSignal(str)

    def __init__(self, camera_source=0, session_id: int = None, parent=None):
        super().__init__(parent)
        self.camera_source = camera_source
        self.session_id    = session_id
        self._running      = True
        self._paused       = False

    def pause(self):  self._paused = True
    def resume(self): self._paused = False

    def stop(self):
        self._running = False
        self._paused  = False

    def run(self):
        import cv2, time
        from services.face_engine import face_engine
        from services.embedding_cache_manager import cache_manager
        from services.attendance_service import attendance_service

        # Mở camera
        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            self.error_occurred.emit(f"Không mở được camera (source={self.camera_source})")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        frame_skip = 0
        stats_timer = time.time()

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            if self._paused:
                # Vẫn hiển thị frame nhưng không xử lý AI
                self.frame_ready.emit(frame, 0)
                time.sleep(0.033)
                continue

            frame_skip += 1
            if frame_skip < 2:
                # Hiển thị mọi frame nhưng chỉ xử lý AI mỗi 2 frame
                self.frame_ready.emit(frame, 0)
                continue
            frame_skip = 0

            t0 = time.perf_counter()
            cache = cache_manager.get_cache()

            if cache.is_empty:
                annotated = frame.copy()
                import cv2 as _cv2
                _cv2.putText(annotated, "Cache trong: chua co hoc vien",
                             (20, 40), _cv2.FONT_HERSHEY_SIMPLEX,
                             0.7, (60, 180, 255), 2)
                self.frame_ready.emit(annotated, 0)
                time.sleep(0.1)
                continue

            # AI Pipeline
            results, elapsed_ms = face_engine.process_frame(frame, cache)
            annotated = face_engine.draw_results(frame, results, elapsed_ms)
            self.frame_ready.emit(annotated, elapsed_ms)

            # Điểm danh
            if attendance_service.is_active:
                events = attendance_service.process_frame_results(
                    results, frame, camera_id=1
                )
                for event in events:
                    self.attendance_done.emit({
                        "student_id":    event.student_id,
                        "student_code":  event.student_code,
                        "full_name":     event.full_name,
                        "time_str":      event.time_str,
                        "similarity":    event.similarity,
                        "snapshot_path": event.snapshot_path,
                    })

            # Stats mỗi 3 giây
            if time.time() - stats_timer >= 3:
                if attendance_service.is_active:
                    self.stats_updated.emit(attendance_service.get_stats())
                stats_timer = time.time()

        cap.release()
        logger.info("AttendanceWorker stopped")


# ─────────────────────────────────────────────
#  AttendanceListItem — 1 dòng học viên có mặt
# ─────────────────────────────────────────────
class AttendanceListItem(QWidget):
    def __init__(self, event: dict, index: int, parent=None):
        super().__init__(parent)
        self.setFixedHeight(60)
        bg = Colors.BG_CARD if index % 2 == 0 else Colors.BG_PANEL
        self.setStyleSheet(f"""
            background: {bg};
            border-bottom: 1px solid {Colors.BORDER};
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(12)

        # Avatar số thứ tự
        idx_lbl = QLabel(str(index))
        idx_lbl.setFixedSize(32, 32)
        idx_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        idx_lbl.setStyleSheet(f"""
            background: {Colors.CYAN}18;
            color: {Colors.CYAN};
            border: 1px solid {Colors.CYAN}44;
            border-radius: 16px;
            font-size: 13px;
            font-weight: 800;
        """)
        layout.addWidget(idx_lbl)

        # Thông tin
        info_col = QVBoxLayout()
        info_col.setSpacing(2)

        name_lbl = QLabel(event["full_name"])
        name_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {Colors.TEXT};"
        )
        code_lbl = QLabel(event["student_code"])
        code_lbl.setStyleSheet(
            f"font-size: 11px; color: {Colors.TEXT_DIM};"
        )
        info_col.addWidget(name_lbl)
        info_col.addWidget(code_lbl)
        layout.addLayout(info_col, 1)

        # Score + Time
        right_col = QVBoxLayout()
        right_col.setSpacing(2)
        right_col.setAlignment(Qt.AlignmentFlag.AlignRight)

        score_pct = event["similarity"] * 100
        score_color = Colors.GREEN if score_pct >= 80 else Colors.ORANGE
        score_lbl = QLabel(f"{score_pct:.1f}%")
        score_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        score_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 800; color: {score_color};"
        )
        time_lbl = QLabel(event["time_str"])
        time_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        time_lbl.setStyleSheet(
            f"font-size: 11px; color: {Colors.TEXT_DIM};"
        )
        right_col.addWidget(score_lbl)
        right_col.addWidget(time_lbl)
        layout.addLayout(right_col)


# ─────────────────────────────────────────────
#  AttendancePage
# ─────────────────────────────────────────────
class AttendancePage(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: AttendanceWorker | None = None
        self._session_id: int | None = None
        self._attendance_count = 0
        self._session_start: datetime | None = None

        self._setup_ui()

        # Đồng hồ cập nhật mỗi giây
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)

    # ─── UI Setup ─────────────────────────────

    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background: transparent; width: 8px; }")

        # LEFT: Camera
        left = self._build_camera_panel()
        splitter.addWidget(left)

        # RIGHT: Control + List
        right = self._build_control_panel()
        splitter.addWidget(right)

        splitter.setSizes([820, 380])
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

    def _build_camera_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background: {Colors.CAM_BG};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 8, 16)
        layout.setSpacing(10)

        # Header camera
        cam_header = QHBoxLayout()
        self._lbl_cam_title = QLabel("📷  Live Camera")
        self._lbl_cam_title.setStyleSheet(
            f"font-size: 14px; font-weight: 700; color: {Colors.TEXT};"
        )
        self._lbl_fps = QLabel("— ms")
        self._lbl_fps.setStyleSheet(
            f"font-size: 12px; color: {Colors.TEXT_DIM}; "
            f"background: {Colors.BG_CARD}; border-radius: 6px; padding: 3px 10px;"
        )
        self._lbl_cam_status = QLabel("⬤  Chờ")
        self._lbl_cam_status.setStyleSheet(
            f"font-size: 12px; color: {Colors.TEXT_DARK}; font-weight: 700;"
        )
        cam_header.addWidget(self._lbl_cam_title)
        cam_header.addStretch()
        cam_header.addWidget(self._lbl_fps)
        cam_header.addWidget(self._lbl_cam_status)
        layout.addLayout(cam_header)

        # Camera preview (lớn)
        self._camera_view = CameraPreviewWidget(
            placeholder_text="📷  Nhấn 'Bắt Đầu Điểm Danh' để mở camera"
        )
        self._camera_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._camera_view, 1)

        # Notification toast (hiện khi nhận ra)
        self._toast = QLabel("")
        self._toast.setFixedHeight(50)
        self._toast.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._toast.setStyleSheet(f"""
            background: {Colors.GREEN}18;
            color: {Colors.GREEN};
            border: 1px solid {Colors.GREEN}44;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 700;
            letter-spacing: 0.5px;
        """)
        self._toast.hide()
        layout.addWidget(self._toast)

        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(self._toast.hide)

        return panel

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"""
            background: {Colors.BG_PANEL};
            border-left: 1px solid {Colors.BORDER};
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)
        panel.setFixedWidth(380)

        # ── Tiêu đề ──
        title = QLabel("Điểm Danh")
        title.setStyleSheet(
            f"font-size: 20px; font-weight: 900; color: {Colors.TEXT};"
        )
        layout.addWidget(title)

        # ── Chọn buổi học ──
        session_card = QWidget()
        session_card.setStyleSheet(card_style(Colors.BORDER))
        sc_layout = QVBoxLayout(session_card)
        sc_layout.setSpacing(10)
        sc_layout.setContentsMargins(14, 12, 14, 12)

        sc_title = QLabel("THIẾT LẬP BUỔI HỌC")
        sc_title.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {Colors.TEXT_DIM}; letter-spacing: 1.5px;"
        )
        sc_layout.addWidget(sc_title)

        # Chọn lớp
        cls_lbl = QLabel("Lớp học")
        cls_lbl.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 12px; font-weight: 600;")
        self._cmb_class = QComboBox()
        self._cmb_class.setStyleSheet(self._combo_style())
        sc_layout.addWidget(cls_lbl)
        sc_layout.addWidget(self._cmb_class)

        # Buổi học — 4 lựa chọn, tự động chọn theo giờ thực
        subj_lbl = QLabel("Buổi học")
        subj_lbl.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 12px; font-weight: 600;")
        self._inp_subject = QComboBox()
        self._inp_subject.addItems([
            "🌅  Buổi sáng  (0h – 9h)",
            "☀️  Buổi trưa  (9h – 15h)",
            "🌤  Buổi chiều  (15h – 21h)",
            "🌙  Buổi tối  (21h – 24h)",
        ])
        self._inp_subject.setStyleSheet(self._combo_style())
        # Tự động chọn buổi theo giờ hiện tại
        self._auto_select_session()
        sc_layout.addWidget(subj_lbl)
        sc_layout.addWidget(self._inp_subject)

        # Ngày điểm danh — mặc định hôm nay, có thể đổi
        date_lbl = QLabel("Ngày điểm danh")
        date_lbl.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 12px; font-weight: 600;")
        self._date_picker = QDateEdit()
        self._date_picker.setDate(QDate.currentDate())
        self._date_picker.setCalendarPopup(True)
        self._date_picker.setDisplayFormat("dd/MM/yyyy")
        self._date_picker.setStyleSheet(f"""
            QDateEdit {{
                background: {Colors.BG_INPUT};
                color: {Colors.TEXT};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 13px;
            }}
            QDateEdit:focus {{ border-color: {Colors.CYAN}; }}
            QDateEdit::drop-down {{
                border: none;
                width: 28px;
            }}
            QDateEdit::down-arrow {{
                image: none;
                width: 0;
            }}
            QCalendarWidget QAbstractItemView {{
                background: {Colors.BG_CARD};
                color: {Colors.TEXT};
                selection-background-color: {Colors.CYAN};
                selection-color: #ffffff;
            }}
            QCalendarWidget QToolButton {{
                background: {Colors.BG_CARD};
                color: {Colors.TEXT};
                border: none;
                padding: 4px 8px;
                font-weight: 700;
            }}
            QCalendarWidget QToolButton:hover {{
                background: {Colors.BG_HOVER};
            }}
            QCalendarWidget QWidget#qt_calendar_navigationbar {{
                background: {Colors.BG_PANEL};
            }}
            QCalendarWidget QWidget {{
                background: {Colors.BG_CARD};
                color: {Colors.TEXT};
            }}
        """)
        sc_layout.addWidget(date_lbl)
        sc_layout.addWidget(self._date_picker)

        layout.addWidget(session_card)

        # ── Nút Bắt đầu / Kết thúc ──
        self._btn_start = QPushButton("▶  Bắt Đầu Điểm Danh")
        self._btn_start.setFixedHeight(46)
        self._btn_start.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.GREEN};
                color: #ffffff;
                border: none;
                border-radius: 10px;
                font-size: 15px;
                font-weight: 800;
            }}
            QPushButton:hover {{ background: {Colors.GREEN_DIM}; }}
            QPushButton:disabled {{
                background: {Colors.BORDER};
                color: {Colors.TEXT_DARK};
            }}
        """)
        self._btn_start.clicked.connect(self._start_session)
        layout.addWidget(self._btn_start)

        self._btn_stop = QPushButton("⏹  Kết Thúc")
        self._btn_stop.setFixedHeight(46)
        self._btn_stop.setEnabled(False)
        self._btn_stop.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.RED};
                color: #ffffff;
                border: none;
                border-radius: 10px;
                font-size: 15px;
                font-weight: 800;
            }}
            QPushButton:hover {{ background: {Colors.RED_DIM}; }}
            QPushButton:disabled {{
                background: {Colors.BORDER};
                color: {Colors.TEXT_DARK};
            }}
        """)
        self._btn_stop.clicked.connect(self._stop_session)
        layout.addWidget(self._btn_stop)

        # ── Thống kê realtime ──
        stats_card = QWidget()
        stats_card.setStyleSheet(card_style(Colors.BORDER))
        stats_layout = QGridLayout(stats_card)
        stats_layout.setSpacing(10)
        stats_layout.setContentsMargins(14, 12, 14, 12)

        def stat_cell(label: str, color: str):
            col = QVBoxLayout()
            val = QLabel("—")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val.setStyleSheet(
                f"font-size: 24px; font-weight: 900; color: {color};"
            )
            lbl = QLabel(label)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(f"font-size: 11px; color: {Colors.TEXT_DIM};")
            col.addWidget(val)
            col.addWidget(lbl)
            return col, val

        c1, self._stat_present = stat_cell("Có mặt",  Colors.GREEN)
        c2, self._stat_absent  = stat_cell("Vắng",     Colors.RED)
        c3, self._stat_total   = stat_cell("Tổng HV",  Colors.CYAN)
        c4, self._stat_elapsed = stat_cell("Thời gian",Colors.ORANGE)

        stats_layout.addLayout(c1, 0, 0)
        stats_layout.addLayout(c2, 0, 1)
        stats_layout.addLayout(c3, 0, 2)
        stats_layout.addLayout(c4, 0, 3)
        layout.addWidget(stats_card)

        # ── Danh sách có mặt (scroll) ──
        list_header = QHBoxLayout()
        list_title = QLabel("DANH SÁCH CÓ MẶT")
        list_title.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {Colors.TEXT_DIM}; letter-spacing: 1.5px;"
        )
        self._lbl_list_count = QLabel("0")
        self._lbl_list_count.setStyleSheet(
            f"font-size: 12px; font-weight: 800; color: {Colors.CYAN}; "
            f"background: {Colors.CYAN}18; border-radius: 10px; padding: 1px 8px;"
        )
        list_header.addWidget(list_title)
        list_header.addStretch()
        list_header.addWidget(self._lbl_list_count)
        layout.addLayout(list_header)

        # Scroll area cho danh sách
        self._list_scroll = QScrollArea()
        self._list_scroll.setWidgetResizable(True)
        self._list_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {Colors.BORDER};
                border-radius: 10px;
                background: {Colors.BG_PANEL};
            }}
            QScrollBar:vertical {{
                background: {Colors.BG_PANEL};
                width: 5px;
                border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: {Colors.BORDER_LT};
                border-radius: 3px;
            }}
        """)

        self._list_container = QWidget()
        self._list_container.setStyleSheet(f"background: {Colors.BG_PANEL};")
        self._list_layout = QVBoxLayout(self._list_container)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(0)
        self._list_layout.addStretch()

        self._list_scroll.setWidget(self._list_container)
        layout.addWidget(self._list_scroll, 1)

        # Đồng hồ phiên
        self._lbl_session_clock = QLabel("00:00:00")
        self._lbl_session_clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_session_clock.setStyleSheet(
            f"font-size: 13px; color: {Colors.TEXT_DARK}; font-weight: 700; letter-spacing: 1px;"
        )
        layout.addWidget(self._lbl_session_clock)

        # Load classes
        self._load_classes()
        return panel

    # ─── Logic ────────────────────────────────

    def _auto_select_session(self):
        """Tự động chọn buổi học dựa trên giờ hiện tại."""
        hour = datetime.now().hour
        if hour < 9:
            idx = 0   # Buổi sáng  0h–9h
        elif hour < 15:
            idx = 1   # Buổi trưa  9h–15h
        elif hour < 21:
            idx = 2   # Buổi chiều 15h–21h
        else:
            idx = 3   # Buổi tối   21h–24h
        self._inp_subject.setCurrentIndex(idx)

    def _load_classes(self):
        self._cmb_class.clear()
        self._cmb_class.addItem("-- Chọn lớp --", None)
        try:
            from database.repositories import class_repo
            for cls in class_repo.get_all():
                self._cmb_class.addItem(
                    f"{cls.class_code} — {cls.class_name}", cls.class_id
                )
        except Exception as e:
            logger.warning(f"Không load lớp: {e}")

    def _start_session(self):
        class_id = self._cmb_class.currentData()

        # Lấy ngày từ date picker
        qdate        = self._date_picker.date()
        session_date = date(qdate.year(), qdate.month(), qdate.day())

        if not class_id:
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng chọn lớp học!")
            return

        try:
            from services.attendance_service import attendance_service
            from services.face_engine import face_engine

            if not face_engine.is_ready:
                QMessageBox.warning(self, "Model chưa sẵn sàng",
                                    "Model AI đang load. Vui lòng đợi vài giây!")
                return

            # Tạo và bắt đầu session
            sid = attendance_service.create_session(
                class_id=class_id,
                session_date=session_date,
            )
            attendance_service.start_session(sid)
            self._session_id    = sid
            self._session_start = datetime.now()
            self._attendance_count = 0

            # Callback khi có học viên điểm danh
            attendance_service.on_attendance = self._on_attendance_event

            # Khởi động worker
            self._worker = AttendanceWorker(camera_source=0, session_id=sid)
            self._worker.frame_ready.connect(self._on_frame)
            self._worker.attendance_done.connect(self._on_attendance_done)
            self._worker.stats_updated.connect(self._update_stats_ui)
            self._worker.error_occurred.connect(self._on_camera_error)
            self._worker.start()

            # Cập nhật UI
            self._btn_start.setEnabled(False)
            self._btn_stop.setEnabled(True)
            self._cmb_class.setEnabled(False)
            self._inp_subject.setEnabled(False)
            self._date_picker.setEnabled(False)
            self._set_cam_status("● Đang điểm danh", Colors.GREEN)
            self._stat_present.setText("0")
            self._stat_absent.setText("—")
            self._stat_total.setText("—")

            logger.success(f"Bắt đầu điểm danh session={sid}")

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể bắt đầu: {e}")
            logger.error(f"Start session error: {e}")

    def _stop_session(self):
        reply = QMessageBox.question(
            self, "Xác nhận kết thúc",
            "Bạn có chắc muốn kết thúc buổi điểm danh?\n"
            "Các học viên chưa điểm danh sẽ được đánh dấu VẮNG.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Dừng worker
        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None

        # Kết thúc session
        try:
            from services.attendance_service import attendance_service
            session = attendance_service.end_session()
            if session:
                QMessageBox.information(
                    self, "Buổi điểm danh đã kết thúc",
                    f"📊 Tổng kết:\n\n"
                    f"  ✅ Có mặt:  {session.present_count} học viên\n"
                    f"  ❌ Vắng:    {session.absent_count} học viên\n"
                    f"  📚 Tổng:    {session.present_count + session.absent_count} học viên\n\n"
                    f"  ⏱ Thời gian: {self._get_elapsed()}\n\n"
                    f"Vào tab 'Báo Cáo' để xuất Excel/PDF."
                )
        except Exception as e:
            logger.error(f"Stop session error: {e}")

        # Reset UI
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._cmb_class.setEnabled(True)
        self._inp_subject.setEnabled(True)
        self._date_picker.setEnabled(True)
        self._date_picker.setDate(QDate.currentDate())   # reset về hôm nay
        self._auto_select_session()                       # re-auto-select buổi
        self._camera_view.clear()
        self._set_cam_status("⬤  Chờ", Colors.TEXT_DARK)
        self._session_id    = None
        self._session_start = None
        self._lbl_session_clock.setText("00:00:00")
        self._lbl_fps.setText("— ms")
        logger.info("Session stopped")

    def _on_attendance_event(self, event):
        pass  # Xử lý qua signal từ worker

    # ─── Signals từ Worker ────────────────────

    def _on_frame(self, frame: np.ndarray, elapsed_ms: float):
        self._camera_view.update_frame(frame)
        if elapsed_ms > 0:
            color = Colors.GREEN if elapsed_ms < 150 else \
                    Colors.ORANGE if elapsed_ms < 300 else Colors.RED
            self._lbl_fps.setText(f"{elapsed_ms:.0f}ms")
            self._lbl_fps.setStyleSheet(
                f"font-size: 12px; color: {color}; "
                f"background: {Colors.BG_CARD}; border-radius: 6px; padding: 3px 10px;"
            )

    def _on_attendance_done(self, event: dict):
        """Thêm học viên vào danh sách + hiện toast."""
        self._attendance_count += 1

        # Thêm vào danh sách
        item = AttendanceListItem(event, self._attendance_count)
        # Chèn trước stretch (index cuối - 1)
        count = self._list_layout.count()
        self._list_layout.insertWidget(count - 1, item)

        # Cập nhật số đếm
        self._lbl_list_count.setText(str(self._attendance_count))
        self._stat_present.setText(str(self._attendance_count))

        # Auto scroll xuống cuối
        QTimer.singleShot(50, lambda: self._list_scroll.verticalScrollBar().setValue(
            self._list_scroll.verticalScrollBar().maximum()
        ))

        # Toast notification
        self._show_toast(
            f"✅  {event['full_name']}  —  {event['time_str']}  ({event['similarity']*100:.1f}%)"
        )
        logger.info(f"Điểm danh: {event['full_name']} @ {event['time_str']}")

    def _update_stats_ui(self, stats: dict):
        if "total_recorded" in stats:
            self._stat_present.setText(str(stats["total_recorded"]))
        if "in_cooldown" in stats:
            pass  # có thể hiển thị

    def _on_camera_error(self, msg: str):
        self._set_cam_status(f"⬤  Lỗi: {msg[:40]}", Colors.RED)
        logger.error(f"Camera error: {msg}")

    # ─── UI helpers ───────────────────────────

    def _show_toast(self, msg: str):
        self._toast.setText(msg)
        self._toast.show()
        self._toast_timer.start(3000)

    def _set_cam_status(self, text: str, color: str):
        self._lbl_cam_status.setText(text)
        self._lbl_cam_status.setStyleSheet(
            f"font-size: 12px; color: {color}; font-weight: 700;"
        )

    def _update_clock(self):
        if self._session_start:
            delta   = datetime.now() - self._session_start
            total_s = int(delta.total_seconds())
            h = total_s // 3600
            m = (total_s % 3600) // 60
            s = total_s % 60
            self._lbl_session_clock.setText(f"{h:02d}:{m:02d}:{s:02d}")
            self._stat_elapsed.setText(f"{m:02d}:{s:02d}" if h == 0 else f"{h}h{m:02d}m")
            self._lbl_session_clock.setStyleSheet(
                f"font-size: 13px; color: {Colors.CYAN}; font-weight: 700; letter-spacing: 1px;"
            )

    def _get_elapsed(self) -> str:
        if not self._session_start:
            return "—"
        delta   = datetime.now() - self._session_start
        total_s = int(delta.total_seconds())
        m, s    = divmod(total_s, 60)
        h, m    = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _combo_style(self) -> str:
        return f"""
            QComboBox {{
                background: {Colors.BG_INPUT};
                color: {Colors.TEXT};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 13px;
            }}
            QComboBox:focus {{ border-color: {Colors.CYAN}; }}
            QComboBox QAbstractItemView {{
                background: {Colors.BG_CARD};
                color: {Colors.TEXT};
                selection-background-color: {Colors.BG_HOVER};
            }}
        """

    def closeEvent(self, event):
        if self._worker:
            self._worker.stop()
            self._worker.wait(2000)
        super().closeEvent(event)

    def hideEvent(self, event):
        # Pause khi chuyển tab (không tắt session)
        if self._worker and not self._worker._paused:
            self._worker.pause()
        super().hideEvent(event)

    def showEvent(self, event):
        # Resume khi quay lại tab
        if self._worker and self._worker._paused:
            self._worker.resume()
        self._load_classes()
        super().showEvent(event)