"""
ui/pages/attendance_page.py
"""
import numpy as np
from datetime import datetime, date

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QFrame,
    QScrollArea, QSizePolicy, QMessageBox,
    QSplitter, QDateEdit, QStackedWidget,
    QMenu,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QDate
from PyQt6.QtGui import QFont, QColor, QPixmap, QImage, QAction
from loguru import logger
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.styles.theme import Colors, card_style, combo_style
from ui.widgets.camera_preview import CameraPreviewWidget
from database.repositories import camera_repo, class_repo, record_repo
from config import app_config
import requests


# ─────────────────────────────────────────────
#  Worker: chạy AI pipeline trong QThread
# ─────────────────────────────────────────────
class AttendanceWorker(QThread):
    """
    Thread đọc Camera thuần tuý (không chạy AI).
    Giành nhiệm vụ AI cho Mini PC.
    """
    frame_ready = pyqtSignal(np.ndarray, float)
    error_occurred = pyqtSignal(str)

    def __init__(self, camera_source=0, parent=None):
        super().__init__(parent)
        self.camera_source = camera_source
        self._running = True
        self._paused = False

    def pause(self): self._paused = True
    def resume(self): self._paused = False

    def stop(self):
        self._running = False
        self._paused = False

    def run(self):
        import cv2, time
        from config import camera_config

        source = int(self.camera_source) if str(self.camera_source).isdigit() else self.camera_source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.error_occurred.emit(f"Không mở được camera (source={self.camera_source})")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        last_time = time.time()
        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            now = time.time()
            elapsed_ms = (now - last_time) * 1000
            last_time = now

            if self._paused:
                self.frame_ready.emit(frame, 0)
                time.sleep(0.033)
                continue

            self.frame_ready.emit(frame, elapsed_ms)
            
            # Giới hạn FPS cơ bản để không tốn CPU Server
            time.sleep(0.033)

        cap.release()
        logger.info("Camera view stopped")


class RemoteStreamWorker(QThread):
    """
    Worker lấy khung hình từ API Server (do Mini PC upload lên).
    Dùng khi giám sát từ xa qua Edge Box.
    """
    frame_ready = pyqtSignal(np.ndarray, float)
    error_occurred = pyqtSignal(str)

    def __init__(self, api_url="http://127.0.0.1:8000/api/system/frame", parent=None):
        super().__init__(parent)
        self.api_url = api_url
        self._running = True
        self._paused = False

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._running = False

    def run(self):
        import requests, cv2, time
        import numpy as np

        logger.info(f"RemoteStreamWorker started: {self.api_url}")
        last_time = time.time()
        
        while self._running:
            if self._paused:
                time.sleep(0.5)
                continue
            try:
                # Poll khung hình mới nhất từ API
                resp = requests.get(self.api_url, timeout=2)
                if resp.status_code == 200:
                    image_bytes = resp.content
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        now = time.time()
                        elapsed_ms = (now - last_time) * 1000
                        last_time = now
                        self.frame_ready.emit(frame, elapsed_ms)
                    else:
                        time.sleep(0.1)
                else:
                    # Nếu chưa có frame hoặc lỗi, đợi lâu hơn chút
                    time.sleep(0.5)
                
                # Giới hạn tốc độ poll (~30+ FPS max)
                time.sleep(0.03)
            except Exception:
                time.sleep(1) # Lỗi mạng, đợi 1s

        logger.info("RemoteStreamWorker stopped")



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

        display_name = f"{event['full_name']} - {event['class_code']}" if event.get("class_code") else event["full_name"]
        name_lbl = QLabel(display_name)
        name_lbl.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {Colors.TEXT}; border: none; background: transparent;")
        
        detail_lbl = QLabel(f"{event['student_code']}")
        detail_lbl.setStyleSheet(f"font-size: 12px; color: {Colors.TEXT_DIM}; border: none; background: transparent;")
        
        info_col.addWidget(name_lbl)
        info_col.addWidget(detail_lbl)
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
        self._rendered_student_codes = set()

        self._setup_ui()

        # Database polling timer for live sync
        self._db_poll_timer = QTimer(self)
        self._db_poll_timer.timeout.connect(self._poll_live_records)

        # Đồng hồ cập nhật mỗi giây
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)

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
        splitter.setStretchFactor(0, 1) # Cho phép camera giãn nhiều nhất
        splitter.setStretchFactor(1, 0) # Panel phải giữ width cơ bản nhưng vẫn được co giãn nếu cần
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

        # ── Toolbar Chọn Camera (Chỉ hiện khi đang điểm danh) ──
        self._camera_selector_widget = QWidget()
        selector_layout = QHBoxLayout(self._camera_selector_widget)
        selector_layout.setContentsMargins(10, 5, 10, 5)
        selector_layout.setSpacing(15)
        self._camera_selector_widget.setStyleSheet(f"""
            QWidget#Selector {{ 
                background: {Colors.BG_CARD}; border-bottom: 1px solid {Colors.BORDER}; 
            }}
        """)
        self._camera_selector_widget.setObjectName("Selector")

        self._btn_select_camera = QPushButton("📍 CHỌN CAMERA GIÁM SÁT ")
        self._btn_select_camera.setFixedHeight(46)
        self._btn_select_camera.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_select_camera.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_DARK}; color: {Colors.CYAN};
                border: 1.5px solid {Colors.CYAN}44; border-radius: 12px;
                padding: 0 20px; font-size: 14px; font-weight: 800;
            }}
            QPushButton:hover {{ 
                background: {Colors.CYAN}22; border-color: {Colors.CYAN}; 
            }}
            QPushButton:menu-indicator {{ image: none; }}
        """)

        self._lbl_selected_camera = QLabel("📷 Camera: ...")
        self._lbl_selected_camera.setFixedHeight(46)
        self._lbl_selected_camera.setStyleSheet(f"""
            QLabel {{
                background: rgba(59, 130, 246, 0.15); color: #3B82F6;
                border: 1.5px solid rgba(59, 130, 246, 0.4); border-radius: 12px;
                padding: 0 16px; font-size: 13px; font-weight: 700;
            }}
        """)

        selector_layout.addWidget(self._btn_select_camera)
        selector_layout.addWidget(self._lbl_selected_camera)
        selector_layout.addStretch()
        
        self._camera_selector_widget.hide() # Ẩn mặc định
        layout.addWidget(self._camera_selector_widget)

        self._cam_stack = QStackedWidget()
        self._cam_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._cam_stack, 1)

        # Trạng thái 0: Setup (Chỉ có nút Bắt đầu chính giữa)
        self._cam_setup_page = QWidget()
        setup_container = QGridLayout(self._cam_setup_page)
        
        self._btn_start = QPushButton("▶  BẮT ĐẦU ĐIỂM DANH")
        self._btn_start.setFixedSize(380, 100)
        self._btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_start.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.GREEN}; color: white;
                border: 4px solid rgba(255, 255, 255, 0.2); border-radius: 50px;
                font-size: 24px; font-weight: 900; letter-spacing: 2px;
            }}
            QPushButton:hover {{ 
                background: {Colors.GREEN_DIM}; border-color: white;
            }}
        """)
        self._btn_start.clicked.connect(self._start_session)
        setup_container.addWidget(self._btn_start, 0, 0, Qt.AlignmentFlag.AlignCenter)
        
        self._cam_stack.addWidget(self._cam_setup_page)

        # Trạng thái 1: Camera Preview 
        self._camera_view = CameraPreviewWidget(
            placeholder_text="Đang kết nối camera..."
        )
        self._cam_stack.addWidget(self._camera_view)
        
        self._cam_stack.setCurrentIndex(0)

        # Notification toast
        self._toast = QLabel("")
        self._toast.setFixedHeight(60)
        self._toast.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._toast.setStyleSheet(f"background: {Colors.GREEN}18; color: {Colors.GREEN}; border: 2px solid {Colors.GREEN}; border-radius: 12px; font-size: 16px; font-weight: 800;")
        self._toast.hide()
        layout.addWidget(self._toast)
        
        self._selected_camera_source = None
        self._camera_menu = None

        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(self._toast.hide)

        return panel

    def _on_camera_selected(self, source, label):
        """Xử lý khi người dùng chọn camera (Lúc setup hoặc đang live)."""
        self._selected_camera_source = source
        self._lbl_selected_camera.setText(f"📍 Camera: {label}")
        self._lbl_selected_camera.setStyleSheet(f"color: {Colors.CYAN}; font-size: 13px; font-weight: 700;")
        
        # Nếu đang live, chuyển ngay camera mới
        if hasattr(self, "_worker") and self._worker and self._worker.isRunning():
            logger.info(f"🔄 Switching preview to: {label}")
            self._worker.stop()
            self._worker.wait()
            
            is_mini_pc = "MINI PC" in label.upper()
            if is_mini_pc:
                self._worker = RemoteStreamWorker()
            else:
                self._worker = AttendanceWorker(camera_source=source)
                
            self._worker.frame_ready.connect(self._on_frame)
            self._worker.error_occurred.connect(self._on_camera_error)
            self._worker.start()
            
        self._btn_start.setEnabled(True)
        self._show_toast(f"🔄 Đang kết nối tới {label}...")

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"""
            background: {Colors.BG_PANEL};
            border-left: 1px solid {Colors.BORDER};
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        panel.setMinimumWidth(380)

        # ── Tiêu đề ──
        title = QLabel("Điểm Danh")
        title.setStyleSheet(f"font-size: 22px; font-weight: 900; color: {Colors.TEXT};")
        layout.addWidget(title)

        # ── Chọn buổi học ──
        session_card = QWidget()
        session_card.setStyleSheet(card_style(Colors.BORDER, radius=12))
        sc_layout = QVBoxLayout(session_card)
        sc_layout.setSpacing(6)
        sc_layout.setContentsMargins(16, 16, 16, 16)

        sc_title = QLabel("THIẾT LẬP BUỔI HỌC")
        sc_title.setStyleSheet(
            f"font-size: 11px; font-weight: 800; color: {Colors.TEXT_DIM}; letter-spacing: 1.5px; border: none; background: transparent;"
        )
        sc_layout.addWidget(sc_title)

        # Chọn lớp đã được gỡ bỏ theo yêu cầu

        self._refresh_cameras()

        # Buổi học
        subj_lbl = QLabel("Buổi học")
        subj_lbl.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 12px; font-weight: 600; border: none; background: transparent; padding-top: 6px;")
        self._inp_subject = QComboBox()
        self._inp_subject.addItems([
            "🌅 Sáng (0h – 9h)",
            "☀️ Trưa (9h – 15h)",
            "🌤 Chiều (15h – 21h)",
            "🌙 Tối (21h – 24h)",
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
                padding: 8px 12px;
                font-size: 13px;
                min-height: 36px;
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
        row_dt.addWidget(self._inp_subject, 3)
        row_dt.addWidget(self._date_picker, 2)
        
        sc_layout.addWidget(subj_lbl)
        sc_layout.addLayout(row_dt)
        sc_layout.addStretch()
        layout.addWidget(session_card)

        # ── Nút Bắt đầu / Kết thúc ──
        # Nút START đã được move sang panel bên trái

        self._btn_stop = QPushButton("⏹  KẾT THÚC")
        self._btn_stop.setFixedHeight(44)
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
        stats_layout.setContentsMargins(12, 12, 12, 12)

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

        return panel

    # ─── Logic ────────────────────────────────

    def _auto_select_session(self):
        hour = datetime.now().hour
        if hour < 9: idx = 0
        elif hour < 15: idx = 1
        elif hour < 21: idx = 2
        else: idx = 3
        self._inp_subject.setCurrentIndex(idx)

    def _refresh_cameras(self):
        """Tải danh sách camera và xây dựng Menu phân cấp đẹp mắt."""
        try:
            # 1. Lấy dữ liệu từ DB
            cameras = camera_repo.get_all(active_only=True)
            
            # 2. Xây dựng cấu trúc mapping: { "E4": { "Tầng 1": "rtsp://..." } }
            cam_map = {}
            special_cams = [] # Dành cho Mini PC,...
            
            for cam in cameras:
                name = cam.camera_name.upper().strip()
                source = cam.rtsp_url
                
                # Check Mini PC (Ư u tiên hàng đầu)
                if "MINI PC" in name:
                    special_cams.append((cam.camera_name, source))
                    continue

                # Tìm Building (E1-E6)
                building = None
                for b in ["E1", "E2", "E3", "E4", "E5", "E6"]:
                    if b in name:
                        building = b
                        break
                
                # Tìm Tầng (1-5)
                floor = None
                for f in range(1, 6):
                    if f"TẦNG {f}" in name or f"TANG {f}" in name or f"F{f}" in name:
                        floor = f"Tầng {f}"
                        break
                
                if building and floor:
                    if building not in cam_map: cam_map[building] = {}
                    cam_map[building][floor] = source
                else:
                    # Nếu không khớp quy luật E1-E6/Tầng, cho vào danh sách Đặc biệt
                    special_cams.append((cam.camera_name, source))

            # 3. Tạo Menu
            main_menu = QMenu(self)
            main_menu.setCursor(Qt.CursorShape.PointingHandCursor)
            main_menu.setStyleSheet(f"""
                QMenu {{
                    background-color: {Colors.BG_CARD};
                    border: 1px solid {Colors.BORDER_LT};
                    border-radius: 12px;
                    padding: 6px;
                }}
                QMenu::item {{
                    padding: 10px 32px;
                    border-radius: 6px;
                    color: {Colors.TEXT};
                    font-size: 14px;
                }}
                QMenu::item:selected {{
                    background-color: {Colors.CYAN}22;
                    color: {Colors.CYAN};
                    font-weight: 700;
                }}
                QMenu::separator {{
                    height: 1px;
                    background: {Colors.BORDER};
                    margin: 4px 8px;
                }}
            """)

            # Thêm các Camera đặc biệt lên đầu (Mini PC, ...)
            for c_name, source in special_cams:
                icon = "🔘" if "MINI" in c_name.upper() else "🎥"
                action = main_menu.addAction(f"{icon}  {c_name}")
                action.triggered.connect(lambda chk, s=source, l=c_name: self._on_camera_selected(s, l))
            
            if special_cams:
                main_menu.addSeparator()

            # Thêm cụm Tòa nhà E1-E6
            for b_idx in range(1, 7):
                b_name = f"E{b_idx}"
                b_menu = main_menu.addMenu(f"🏢  Tòa nhà {b_name}")
                
                is_used = False
                for f_idx in range(1, 6):
                    f_name = f"Tầng {f_idx}"
                    source = cam_map.get(b_name, {}).get(f_name)
                    
                    action_label = f"{f_name} - {b_name}"
                    if source:
                        action = b_menu.addAction(f"🟢  {action_label}")
                        action.triggered.connect(lambda chk, s=source, l=action_label: self._on_camera_selected(s, l))
                        is_used = True
                    else:
                        action = b_menu.addAction(f"⚪  {action_label} (Chưa có)")
                        action.setEnabled(False)
                
                # Nếu tòa nhà này có camera thật, in đậm tên tòa nhà (Optional: sử dụng font hoặc icon khác)
                if is_used:
                    pass 

            self._btn_select_camera.setMenu(main_menu)
            self._camera_menu = main_menu
            
            # Mặc định nhấn Bắt đầu
            self._btn_start.setEnabled(True)
            
            # TỰ ĐỘNG CHỌN MẶC ĐỊNH: 
            # Ưu tiên 1: Mini PC
            # Ưu tiên 2: E4 - Tầng 1
            default_source = None
            default_label = ""
            
            for c_name, source in special_cams:
                if "MINI PC" in c_name.upper():
                    default_source = source
                    default_label = c_name
                    break
            
            if not default_source:
                default_source = cam_map.get("E4", {}).get("Tầng 1")
                default_label = "Tầng 1 - E4"
            if default_source:
                self._selected_camera_source = default_source
                self._lbl_selected_camera.setText(f"📷 Camera: {default_label}")
                self._lbl_selected_camera.setStyleSheet(f"color: {Colors.CYAN}; font-size: 13px; font-weight: 700;")
            else:
                self._lbl_selected_camera.setText("📷 Camera: (Chưa chọn)")
                self._lbl_selected_camera.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 13px; font-weight: 700;")

            
        except Exception as e:
            logger.error(f"Error building camera menu: {e}")
            self._lbl_selected_camera.setText("⚠️ Lỗi tải danh sách Camera")

    def _start_session(self):
        camera_source = self._selected_camera_source
        qdate = self._date_picker.date()
        session_date = date(qdate.year(), qdate.month(), qdate.day())

        try:
            from database.repositories import class_repo
            classes = class_repo.get_all()
            if not classes:
                QMessageBox.warning(self, "Lỗi", "Chưa có danh mục Lớp Học nào trong CSDL!")
                return
            class_id = classes[0].class_id
        except Exception:
            class_id = 1

        # camera_source có thể là None (Chế độ tổng quát)
        pass

        try:
            from services.attendance_service import attendance_service

            # Xoá ds hiển thị cũ
            while self._list_layout.count() > 1:
                child = self._list_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            
            self._rendered_student_codes.clear()

            sid = attendance_service.create_session(
                class_id=class_id, 
                subject_name=self._inp_subject.currentText(), 
                session_date=session_date
            )
            attendance_service.start_session(sid)
            self._session_id = sid
            self._session_start = datetime.now()
            self._attendance_count = 0

            # Bật các timer cập nhật
            self._db_poll_timer.start(1000)
            self._clock_timer.start(1000)

            # Gửi lệnh START tới API Server (Dùng Thread để không lag UI)
            def send_start():
                try:
                    requests.post("http://127.0.0.1:8000/api/system/command", json={
                        "command": "START",
                        "session_id": sid,
                        "class_id": class_id,
                        "target_camera": camera_source
                    }, headers={"X-API-Key": "faceattend_secret_2026"}, timeout=5)
                except: pass
            
            import threading
            threading.Thread(target=send_start, daemon=True).start()

            # KIỂM TRA: Nếu chọn Mini PC, nhường camera cho Mini PC xử lý AI
            is_mini_pc = "MINI PC" in self._lbl_selected_camera.text().upper()
            
            if camera_source and not is_mini_pc:
                self._worker = AttendanceWorker(camera_source=camera_source)
                self._worker.frame_ready.connect(self._on_frame)
                self._worker.error_occurred.connect(self._on_camera_error)
                self._worker.start()
                self._cam_stack.setCurrentIndex(1)
            elif is_mini_pc:
                self._worker = RemoteStreamWorker()
                self._worker.frame_ready.connect(self._on_frame)
                self._worker.error_occurred.connect(self._on_camera_error)
                self._worker.start()
                self._camera_view.set_placeholder("🖥️  ĐANG KẾT NỐI MINI PC...") 
                self._cam_stack.setCurrentIndex(1)
            else:
                self._worker = None
                self._camera_view.set_placeholder("") 
                self._cam_stack.setCurrentIndex(1)

            self._camera_selector_widget.show() # Hiện thanh chọn camera khi đang live
            self._btn_start.setEnabled(False)
            self._cam_stack.setCurrentIndex(1)
            self._btn_stop.setEnabled(True)
            self._btn_select_camera.setEnabled(True)
            self._inp_subject.setEnabled(False)
            self._date_picker.setEnabled(False)
            
            self._set_cam_status("● Đang điểm danh", Colors.GREEN)
            self._stat_present.setText("0")
            self._stat_absent.setText("—")
            self._stat_total.setText("—")
            self._lbl_list_count.setText("0")

        except Exception as e:
            logger.error(f"Start session error: {e}")
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
            
            # Dừng các timer
            self._db_poll_timer.stop()
            self._clock_timer.stop()

            # Ra lệnh STOP qua API (Dùng Thread)
            def send_stop():
                try:
                    requests.post("http://127.0.0.1:8000/api/system/command", json={
                        "command": "STOP"
                    }, headers={"X-API-Key": "faceattend_secret_2026"}, timeout=5)
                except: pass
            
            import threading
            threading.Thread(target=send_stop, daemon=True).start()

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
        self._camera_selector_widget.hide() # Ẩn lại thanh chọn
        self._btn_select_camera.setEnabled(True)
        self._inp_subject.setEnabled(True)
        self._date_picker.setEnabled(True)
        self._camera_view.clear()
        
        self._cam_stack.setCurrentIndex(0)
        self._set_cam_status("⬤  Chờ", Colors.TEXT_DARK)
        self._session_start = None


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
        
        self._show_toast(f"✅  {event['full_name']} - {event['class_code']}  ({event['similarity']*100:.1f}%)")

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
        
        # Đồng bộ trạng thái HUD trên Camera Preview
        if "điểm danh" in text.lower():
            self._camera_view.set_status("ONLINE", color)
        else:
            self._camera_view.set_status("OFFLINE", color)

    def _poll_live_records(self):
        """Websocket/Polling alternative: Fetch records from DB periodically."""
        if not self._session_id: return
        
        try:
            from database.repositories import record_repo
            present_list = record_repo.get_present_list(self._session_id)
            
            for p in present_list:
                code = p["code"]
                if code not in self._rendered_student_codes:
                    self._rendered_student_codes.add(code)
                    
                    event = {
                        "full_name": p["name"],
                        "class_code": "", # Fetching if needed, leaving empty for now
                        "student_code": code,
                        "similarity": p["score"],
                        "time_str": p["time"].strftime("%H:%M:%S") if hasattr(p["time"], "strftime") else str(p["time"])
                    }
                    self._on_attendance_done(event)
                    
            # Update stats
            self._stat_total.setText(str(len(present_list) + 10)) # Just a visual
            
        except Exception as e:
            pass

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
        self._refresh_cameras()
        super().showEvent(event)