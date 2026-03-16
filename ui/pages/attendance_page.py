"""
ui/pages/attendance_page.py
"""
import numpy as np
from datetime import datetime, date

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QFrame,
    QScrollArea, QSizePolicy, QMessageBox,
    QSplitter, QDateEdit,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QDate
from PyQt6.QtGui import QFont, QColor, QPixmap, QImage
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.styles.theme import Colors, card_style, combo_style
from ui.widgets.camera_preview import CameraPreviewWidget
from config import CAMERAS


# ─────────────────────────────────────────────
#  Worker: chạy AI pipeline trong QThread
# ─────────────────────────────────────────────
class AttendanceWorker(QThread):
    """
    Thread chạy vòng lặp: Camera → FaceEngine → AttendanceService.
    Phát signals lên UI thread an toàn.
    """
    frame_ready     = pyqtSignal(np.ndarray, float)   # (annotated_frame, ms)
    attendance_done = pyqtSignal(dict)                # event dict
    stats_updated   = pyqtSignal(dict)                # stats
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

        # Đổi str thành int nếu là webcam USB "0", "1"
        source = int(self.camera_source) if str(self.camera_source).isdigit() else self.camera_source
        
        # Mở camera
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.error_occurred.emit(f"Không mở được camera (source={self.camera_source})")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # IP Camera giảm độ trễ
        if str(self.camera_source).startswith(("rtsp", "http")):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        frame_skip = 0
        stats_timer = time.time()

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            if self._paused:
                self.frame_ready.emit(frame, 0)
                time.sleep(0.033)
                continue

            frame_skip += 1
            if frame_skip < 2:
                self.frame_ready.emit(frame, 0)
                continue
            frame_skip = 0

            t0 = time.perf_counter()
            cache = cache_manager.get_cache()

            if cache.is_empty:
                annotated = frame.copy()
                import cv2 as _cv2
                _cv2.putText(annotated, "Database trong: chua co hoc vien",
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
        self.setFixedHeight(70) # Tăng chiều cao tránh mất chữ
        bg = Colors.BG_CARD if index % 2 == 0 else Colors.BG_PANEL
        self.setStyleSheet(f"""
            QWidget {{
                background: {bg};
                border-bottom: 1px solid {Colors.BORDER};
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(16)

        # Avatar số thứ tự
        idx_lbl = QLabel(str(index))
        idx_lbl.setFixedSize(36, 36)
        idx_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        idx_lbl.setStyleSheet(f"""
            QLabel {{
                background: {Colors.CYAN}18;
                color: {Colors.CYAN};
                border: 1px solid {Colors.CYAN}44;
                border-radius: 18px;
                font-size: 14px;
                font-weight: 800;
            }}
        """)
        layout.addWidget(idx_lbl)

        # Thông tin
        info_col = QVBoxLayout()
        info_col.setSpacing(4)
        info_col.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        name_lbl = QLabel(event["full_name"])
        name_lbl.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {Colors.TEXT}; border: none; background: transparent;")
        
        code_lbl = QLabel(event["student_code"])
        code_lbl.setStyleSheet(f"font-size: 12px; color: {Colors.TEXT_DIM}; border: none; background: transparent;")
        
        info_col.addWidget(name_lbl)
        info_col.addWidget(code_lbl)
        layout.addLayout(info_col, 1)

        # Score + Time
        right_col = QVBoxLayout()
        right_col.setSpacing(4)
        right_col.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)

        score_pct = event["similarity"] * 100
        score_color = Colors.GREEN if score_pct >= 80 else Colors.ORANGE
        score_lbl = QLabel(f"{score_pct:.1f}%")
        score_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        score_lbl.setStyleSheet(f"font-size: 14px; font-weight: 800; color: {score_color}; border: none; background: transparent;")
        
        time_lbl = QLabel(event["time_str"])
        time_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        time_lbl.setStyleSheet(f"font-size: 12px; color: {Colors.TEXT_DIM}; border: none; background: transparent;")
        
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

        splitter.setSizes([850, 420]) # Nới rộng panel bên phải để không bị cắt chữ
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

    def _build_camera_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background: {Colors.CAM_BG};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 10, 20)
        layout.setSpacing(16)

        # Header camera
        cam_header = QHBoxLayout()
        self._lbl_cam_title = QLabel("📷  Live Camera")
        self._lbl_cam_title.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {Colors.TEXT};"
        )
        
        self._lbl_fps = QLabel("— ms")
        self._lbl_fps.setStyleSheet(
            f"font-size: 13px; font-weight: 600; color: {Colors.TEXT_DIM}; "
            f"background: {Colors.BG_CARD}; border-radius: 6px; padding: 4px 12px;"
        )
        
        self._lbl_cam_status = QLabel("⬤  Chờ")
        self._lbl_cam_status.setStyleSheet(
            f"font-size: 13px; color: {Colors.TEXT_DARK}; font-weight: 700;"
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
        self._toast.setFixedHeight(60)
        self._toast.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._toast.setStyleSheet(f"""
            background: {Colors.GREEN}18;
            color: {Colors.GREEN};
            border: 2px solid {Colors.GREEN};
            border-radius: 12px;
            font-size: 16px;
            font-weight: 800;
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
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        panel.setFixedWidth(420)

        # ── Tiêu đề ──
        title = QLabel("Điểm Danh")
        title.setStyleSheet(f"font-size: 22px; font-weight: 900; color: {Colors.TEXT};")
        layout.addWidget(title)

        # ── Chọn buổi học ──
        session_card = QWidget()
        session_card.setStyleSheet(card_style(Colors.BORDER, radius=12))
        sc_layout = QVBoxLayout(session_card)
        sc_layout.setSpacing(12)
        sc_layout.setContentsMargins(16, 16, 16, 16)

        sc_title = QLabel("THIẾT LẬP BUỔI HỌC")
        sc_title.setStyleSheet(
            f"font-size: 11px; font-weight: 800; color: {Colors.TEXT_DIM}; letter-spacing: 1.5px; border: none; background: transparent;"
        )
        sc_layout.addWidget(sc_title)

        # Chọn lớp
        cls_lbl = QLabel("Lớp học")
        cls_lbl.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 13px; font-weight: 600; border: none; background: transparent;")
        self._cmb_class = QComboBox()
        self._cmb_class.setStyleSheet(combo_style())
        sc_layout.addWidget(cls_lbl)
        sc_layout.addWidget(self._cmb_class)

        # Chọn Camera (Từ config)
        cam_lbl = QLabel("Chọn Camera")
        cam_lbl.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 13px; font-weight: 600; border: none; background: transparent;")
        self._cmb_camera = QComboBox()
        self._cmb_camera.setStyleSheet(combo_style())
        for cam in CAMERAS:
            self._cmb_camera.addItem(cam['name'], cam['source'])
        sc_layout.addWidget(cam_lbl)
        sc_layout.addWidget(self._cmb_camera)

        # Buổi học
        subj_lbl = QLabel("Buổi học")
        subj_lbl.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 13px; font-weight: 600; border: none; background: transparent;")
        self._inp_subject = QComboBox()
        self._inp_subject.addItems([
            "🌅  Buổi sáng  (0h – 9h)",
            "☀️  Buổi trưa  (9h – 15h)",
            "🌤  Buổi chiều  (15h – 21h)",
            "🌙  Buổi tối  (21h – 24h)",
        ])
        self._inp_subject.setStyleSheet(combo_style())
        self._auto_select_session()
        
        # Ngày
        self._date_picker = QDateEdit()
        self._date_picker.setDate(QDate.currentDate())
        self._date_picker.setCalendarPopup(True)
        self._date_picker.setDisplayFormat("dd/MM/yyyy")
        self._date_picker.setStyleSheet(f"""
            QDateEdit {{
                background: {Colors.BG_CARD};
                color: {Colors.TEXT};
                border: 1.5px solid {Colors.BORDER_LT};
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 14px;
                min-height: 42px;
            }}
            QDateEdit:focus {{ border-color: {Colors.CYAN}; }}
            QDateEdit::drop-down {{ border: none; width: 30px; }}
            QDateEdit::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {Colors.TEXT_DIM};
                margin-right: 10px;
            }}
        """)
        
        row_dt = QHBoxLayout()
        row_dt.addWidget(self._inp_subject)
        row_dt.addWidget(self._date_picker)
        sc_layout.addWidget(subj_lbl)
        sc_layout.addLayout(row_dt)

        layout.addWidget(session_card)

        # ── Nút Bắt đầu / Kết thúc ──
        self._btn_start = QPushButton("▶  BẮT ĐẦU ĐIỂM DANH")
        self._btn_start.setFixedHeight(50)
        self._btn_start.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.GREEN}; color: #ffffff;
                border: none; border-radius: 10px;
                font-size: 16px; font-weight: 800; letter-spacing: 1px;
            }}
            QPushButton:hover {{ background: {Colors.GREEN_DIM}; }}
            QPushButton:disabled {{ background: {Colors.BORDER}; color: {Colors.TEXT_DARK}; }}
        """)
        self._btn_start.clicked.connect(self._start_session)
        layout.addWidget(self._btn_start)

        self._btn_stop = QPushButton("⏹  KẾT THÚC")
        self._btn_stop.setFixedHeight(50)
        self._btn_stop.setEnabled(False)
        self._btn_stop.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.RED}; color: #ffffff;
                border: none; border-radius: 10px;
                font-size: 16px; font-weight: 800; letter-spacing: 1px;
            }}
            QPushButton:hover {{ background: {Colors.RED_DIM}; }}
            QPushButton:disabled {{ background: {Colors.BORDER}; color: {Colors.TEXT_DARK}; }}
        """)
        self._btn_stop.clicked.connect(self._stop_session)
        layout.addWidget(self._btn_stop)

        # ── Thống kê realtime ──
        stats_card = QWidget()
        stats_card.setStyleSheet(card_style(Colors.BORDER, radius=12))
        stats_layout = QGridLayout(stats_card)
        stats_layout.setSpacing(12)
        stats_layout.setContentsMargins(16, 16, 16, 16)

        def stat_cell(label: str, color: str):
            col = QVBoxLayout()
            val = QLabel("—")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val.setStyleSheet(f"font-size: 26px; font-weight: 900; color: {color}; border: none; background: transparent;")
            lbl = QLabel(label)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(f"font-size: 12px; color: {Colors.TEXT_DIM}; font-weight: 600; border: none; background: transparent;")
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
        list_title.setStyleSheet(f"font-size: 12px; font-weight: 800; color: {Colors.TEXT_DIM}; letter-spacing: 1.5px;")
        
        self._lbl_list_count = QLabel("0")
        self._lbl_list_count.setStyleSheet(f"""
            font-size: 14px; font-weight: 800; color: {Colors.CYAN};
            background: {Colors.CYAN}18; border-radius: 12px; padding: 2px 10px;
        """)
        list_header.addWidget(list_title)
        list_header.addStretch()
        list_header.addWidget(self._lbl_list_count)
        layout.addLayout(list_header)

        # Scroll area
        self._list_scroll = QScrollArea()
        self._list_scroll.setWidgetResizable(True)
        self._list_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {Colors.BORDER_LT};
                border-radius: 12px;
                background: {Colors.BG_CARD};
            }}
            QScrollBar:vertical {{ background: transparent; width: 6px; }}
            QScrollBar::handle:vertical {{ background: {Colors.BORDER_LT}; border-radius: 3px; }}
        """)

        self._list_container = QWidget()
        self._list_container.setStyleSheet(f"background: {Colors.BG_CARD};")
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
            f"font-size: 14px; color: {Colors.TEXT_DARK}; font-weight: 800; letter-spacing: 2px;"
        )
        layout.addWidget(self._lbl_session_clock)

        self._load_classes()
        return panel

    # ─── Logic ────────────────────────────────

    def _auto_select_session(self):
        hour = datetime.now().hour
        if hour < 9: idx = 0
        elif hour < 15: idx = 1
        elif hour < 21: idx = 2
        else: idx = 3
        self._inp_subject.setCurrentIndex(idx)

    def _load_classes(self):
        self._cmb_class.clear()
        self._cmb_class.addItem("-- Chọn lớp học --", None)
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
        camera_source = self._cmb_camera.currentData()
        qdate = self._date_picker.date()
        session_date = date(qdate.year(), qdate.month(), qdate.day())

        if not class_id:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn lớp học!")
            return

        try:
            from services.attendance_service import attendance_service
            from services.face_engine import face_engine

            if not face_engine.is_ready:
                QMessageBox.warning(self, "Đợi chút", "Model AI đang khởi động, vui lòng thử lại sau vài giây.")
                return

            # Xoá ds hiển thị cũ
            while self._list_layout.count() > 1:
                item = self._list_layout.takeAt(0).widget()
                if item: item.deleteLater()

            sid = attendance_service.create_session(
                class_id=class_id, subject_name=self._inp_subject.currentText(), session_date=session_date
            )
            attendance_service.start_session(sid)
            self._session_id    = sid
            self._session_start = datetime.now()
            self._attendance_count = 0

            attendance_service.on_attendance = self._on_attendance_event

            self._worker = AttendanceWorker(camera_source=camera_source, session_id=sid)
            self._worker.frame_ready.connect(self._on_frame)
            self._worker.attendance_done.connect(self._on_attendance_done)
            self._worker.stats_updated.connect(self._update_stats_ui)
            self._worker.error_occurred.connect(self._on_camera_error)
            self._worker.start()

            self._btn_start.setEnabled(False)
            self._btn_stop.setEnabled(True)
            self._cmb_class.setEnabled(False)
            self._cmb_camera.setEnabled(False)
            self._inp_subject.setEnabled(False)
            self._date_picker.setEnabled(False)
            
            self._set_cam_status("● Đang điểm danh", Colors.GREEN)
            self._stat_present.setText("0")
            self._stat_absent.setText("—")
            self._stat_total.setText("—")
            self._lbl_list_count.setText("0")

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể bắt đầu: {e}")

    def _stop_session(self):
        reply = QMessageBox.question(
            self, "Kết thúc",
            "Bạn có chắc muốn kết thúc?\nHọc viên chưa điểm danh sẽ bị đánh vắng.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes: return

        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None

        try:
            from services.attendance_service import attendance_service
            session = attendance_service.end_session()
            if session:
                QMessageBox.information(
                    self, "Tổng kết",
                    f"✅ Có mặt:  {session.present_count}\n"
                    f"❌ Vắng:    {session.absent_count}\n"
                    f"⏱ Thời gian: {self._get_elapsed()}"
                )
        except Exception as e:
            logger.error(f"Stop session error: {e}")

        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._cmb_class.setEnabled(True)
        self._cmb_camera.setEnabled(True)
        self._inp_subject.setEnabled(True)
        self._date_picker.setEnabled(True)
        self._camera_view.clear()
        self._set_cam_status("⬤  Chờ", Colors.TEXT_DARK)
        self._session_start = None

    def _on_attendance_event(self, event):
        pass # Signal handled in _on_attendance_done

    def _on_frame(self, frame: np.ndarray, elapsed_ms: float):
        self._camera_view.update_frame(frame)
        if elapsed_ms > 0:
            color = Colors.GREEN if elapsed_ms < 150 else Colors.ORANGE if elapsed_ms < 300 else Colors.RED
            self._lbl_fps.setText(f"{elapsed_ms:.0f}ms")
            self._lbl_fps.setStyleSheet(
                f"font-size: 13px; font-weight: 700; color: {color}; "
                f"background: {Colors.BG_CARD}; border-radius: 6px; padding: 4px 12px;"
            )

    def _on_attendance_done(self, event: dict):
        self._attendance_count += 1
        item = AttendanceListItem(event, self._attendance_count)
        count = self._list_layout.count()
        self._list_layout.insertWidget(count - 1, item)

        self._lbl_list_count.setText(str(self._attendance_count))
        self._stat_present.setText(str(self._attendance_count))

        QTimer.singleShot(50, lambda: self._list_scroll.verticalScrollBar().setValue(
            self._list_scroll.verticalScrollBar().maximum()
        ))
        
        self._show_toast(f"✅  {event['full_name']}  ({event['similarity']*100:.1f}%)")

    def _update_stats_ui(self, stats: dict):
        if "total_recorded" in stats:
            self._stat_present.setText(str(stats["total_recorded"]))

    def _on_camera_error(self, msg: str):
        self._set_cam_status(f"⬤  Lỗi", Colors.RED)

    def _show_toast(self, msg: str):
        self._toast.setText(msg)
        self._toast.show()
        self._toast_timer.start(3000)

    def _set_cam_status(self, text: str, color: str):
        self._lbl_cam_status.setText(text)
        self._lbl_cam_status.setStyleSheet(f"font-size: 13px; color: {color}; font-weight: 700;")

    def _update_clock(self):
        if self._session_start:
            delta = datetime.now() - self._session_start
            total_s = int(delta.total_seconds())
            h, m, s = total_s // 3600, (total_s % 3600) // 60, total_s % 60
            self._lbl_session_clock.setText(f"{h:02d}:{m:02d}:{s:02d}")
            self._stat_elapsed.setText(f"{m:02d}:{s:02d}" if h == 0 else f"{h}h{m:02d}m")
            self._lbl_session_clock.setStyleSheet(
                f"font-size: 14px; color: {Colors.CYAN}; font-weight: 800; letter-spacing: 2px;"
            )

    def _get_elapsed(self) -> str:
        if not self._session_start: return "—"
        total_s = int((datetime.now() - self._session_start).total_seconds())
        m, s = divmod(total_s, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def closeEvent(self, event):
        if self._worker:
            self._worker.stop()
            self._worker.wait(2000)
        super().closeEvent(event)

    def hideEvent(self, event):
        if self._worker and not self._worker._paused: self._worker.pause()
        super().hideEvent(event)

    def showEvent(self, event):
        if self._worker and self._worker._paused: self._worker.resume()
        self._load_classes()
        super().showEvent(event)