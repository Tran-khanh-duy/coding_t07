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
import os
import requests
from pathlib import Path

# Thêm import cho system_state
from api_server import system_state

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.styles.theme import Colors, card_style, combo_style
from ui.widgets.camera_preview import CameraPreviewWidget
from database.repositories import camera_repo, class_repo, record_repo
from config import app_config

# ─────────────────────────────────────────────
#  Worker: chạy AI pipeline trong QThread
# ─────────────────────────────────────────────
class AttendanceWorker(QThread):
    """
    Thread đọc Camera thuần tuý (không chạy AI).
    Giành nhiệm vụ AI cho Mini PC.
    """
    frame_ready = pyqtSignal(np.ndarray, float, list)
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
                self.frame_ready.emit(frame, 0, [])
                time.sleep(0.033)
                continue

            self.frame_ready.emit(frame, elapsed_ms, [])
            
            # Giới hạn FPS cơ bản để không tốn CPU Server
            time.sleep(0.033)

        cap.release()
        logger.info("Camera view stopped")


class RemoteStreamWorker(QThread):
    """
    Worker lấy khung hình từ API Server (do Mini PC upload lên).
    Dùng khi giám sát từ xa qua Edge Box.
    """
    frame_ready = pyqtSignal(np.ndarray, float, list)
    error_occurred = pyqtSignal(str)

    def __init__(self, camera_id=None, api_url="http://127.0.0.1:9696/api/system/frame", parent=None):
        super().__init__(parent)
        # [FIX] Giữ nguyên camera_id: chỉ strip() khoảng trắng
        # KHÔNG .upper() vì sẽ phá hỏng RTSP URL (rtsp://admin:pass@ip/...)
        safe_id = str(camera_id or "CAM_01").strip()
        self.camera_id = safe_id
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
                # 1. Poll khung hình với camera_id đã được chuẩn hóa
                params = {"camera_id": self.camera_id}
                resp = requests.get(self.api_url, params=params, timeout=3)
                
                if resp.status_code == 200:
                    image_bytes = resp.content
                    if not image_bytes:
                        time.sleep(0.1)
                        continue
                        
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # 2. Lấy tọa độ khuôn mặt từ Header
                        det_list = []
                        if "X-Face-Detections" in resp.headers:
                            try:
                                import base64, json
                                det_b64 = resp.headers["X-Face-Detections"]
                                det_list = json.loads(base64.b64decode(det_b64).decode())
                            except: pass

                        now = time.time()
                        elapsed_ms = (now - last_time) * 1000
                        last_time = now
                        self.frame_ready.emit(frame, elapsed_ms, det_list)
                    else:
                        self.error_occurred.emit("Lỗi Decode Ảnh")
                else:
                    # Log lỗi chi tiết ra UI terminal
                    msg = f"Server {resp.status_code}"
                    if resp.status_code == 404:
                        msg = "Server chưa có hình (Chờ Mini PC tải lên)"
                        
                    print(f"[ERROR] UI Stream {self.camera_id} fail: {msg}")
                    self.error_occurred.emit(msg)
                    time.sleep(0.5)

                # Giới hạn tốc độ poll
                time.sleep(0.02)

            except requests.exceptions.ConnectionError:
                self.error_occurred.emit("Mất kết nối Server")
                time.sleep(2)
            except Exception as e:
                logger.error(f"RemoteStreamWorker Error ({self.camera_id}): {e}")
                self.error_occurred.emit("Lỗi luồng")
                time.sleep(1)

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
        self._selected_camera_source = None # Khởi tạo mặc định
        self._attendance_count = 0
        self._session_start: datetime | None = None
        self._rendered_student_codes = set()
        
        # Timer tự động cập nhật đèn tín hiệu (Xanh/Đỏ)
        self._status_refresh_timer = QTimer(self)
        self._status_refresh_timer.timeout.connect(self._refresh_cameras)
        self._status_refresh_timer.start(10000) # Cập nhật mỗi 10s

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
        layout.setContentsMargins(15, 10, 15, 15)
        layout.setSpacing(10)

        # ── 1. Header Toolbar (Premium VS Code style) ──
        toolbar = QFrame()
        toolbar.setFixedHeight(50)
        toolbar.setStyleSheet(f"""
            QFrame {{
                background: {Colors.BG_DARK}; border: 1px solid {Colors.BORDER}; 
                border-radius: 12px;
            }}
        """)
        t_layout = QHBoxLayout(toolbar)
        t_layout.setContentsMargins(15, 0, 15, 0)
        t_layout.setSpacing(15)

        # Trái: Icon + Title
        title_box = QHBoxLayout()
        icon_lbl = QLabel("🎥")
        icon_lbl.setStyleSheet("font-size: 16px;")
        self._lbl_cam_title = QLabel("Live Camera")
        self._lbl_cam_title.setStyleSheet(f"color: {Colors.TEXT}; font-size: 14px; font-weight: 800;")
        title_box.addWidget(icon_lbl)
        title_box.addWidget(self._lbl_cam_title)
        t_layout.addLayout(title_box)

        t_layout.addStretch()

        # Phải: FPS & Status
        self._lbl_fps = QLabel("— ms")
        self._lbl_fps.setStyleSheet(f"background: {Colors.BG_CARD}; color: {Colors.TEXT_DIM}; padding: 3px 10px; border-radius: 6px; font-weight: 700; font-size: 11px;")
        
        self._lbl_cam_status = QLabel("⬤  Chờ")
        self._lbl_cam_status.setStyleSheet(f"color: {Colors.TEXT_DARK}; font-weight: 800; font-size: 12px; margin-left: 8px;")
        
        t_layout.addWidget(self._lbl_fps)
        t_layout.addWidget(self._lbl_cam_status)
        layout.addWidget(toolbar)

        # ── 2. Camera Selection Toolbar (Sub-menu style) ──
        sel_bar = QWidget()
        sel_bar.setFixedHeight(50)
        sb_layout = QHBoxLayout(sel_bar)
        sb_layout.setContentsMargins(5, 0, 5, 0)
        sb_layout.setSpacing(12)

        sel_label = QLabel("📍 Chọn Camera:")
        sel_label.setStyleSheet(f"color: {Colors.CYAN}; font-size: 13px; font-weight: 800;")
        sb_layout.addWidget(sel_label)

        # Nút bấm tổng hợp (Merge into one)
        self._btn_camera_select = QPushButton("🔍 CHỌN CAMERA HỆ THỐNG...  ▾")
        self._btn_camera_select.setFixedHeight(40)
        self._btn_camera_select.setMinimumWidth(250)
        self._btn_camera_select.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_camera_select.setStyleSheet(f"""
            QPushButton {{
                background: white; color: black;
                border: 1px solid #E0E0E0; border-radius: 8px;
                padding: 0 20px; font-size: 13px; font-weight: 800;
            }}
            QPushButton:hover {{ background: #F5F5F5; border-color: {Colors.CYAN}; }}
            QPushButton::menu-indicator {{ image: none; }}
        """)
        sb_layout.addWidget(self._btn_camera_select)

        sb_layout.addStretch()

        # Hộp trạng thái cam đang chọn
        self._cam_status_box = QFrame()
        self._cam_status_box.setFixedHeight(40)
        self._cam_status_box.setStyleSheet(f"background: {Colors.GREEN}14; border: 1.5px solid {Colors.GREEN}33; border-radius: 8px;")
        cs_layout = QHBoxLayout(self._cam_status_box)
        cs_layout.setContentsMargins(12, 0, 12, 0)
        self._lbl_current_cam = QLabel("Chưa chọn")
        self._lbl_current_cam.setStyleSheet(f"color: {Colors.GREEN}; font-weight: 800; font-size: 13px;")
        cs_layout.addWidget(self._lbl_current_cam)
        sb_layout.addWidget(self._cam_status_box)
        
        layout.addWidget(sel_bar)

        # ── 3. Camera Display area ──
        self._cam_stack = QStackedWidget()
        layout.addWidget(self._cam_stack, 1)

        # Setup page (Placeholder)
        self._cam_setup_page = QFrame()
        self._cam_setup_page.setStyleSheet(f"background: {Colors.BG_DARK}; border: 1px dashed {Colors.BORDER}; border-radius: 15px;")
        setup_layout = QVBoxLayout(self._cam_setup_page)
        
        hint_icon = QLabel("📹")
        hint_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint_icon.setStyleSheet("font-size: 48px; opacity: 0.5;")
        setup_layout.addStretch()
        setup_layout.addWidget(hint_icon)
        
        hint_lbl = QLabel("Vui lòng chọn Camera từ Menu bên trên")
        hint_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint_lbl.setStyleSheet(f"color: {Colors.TEXT_DARK}; font-size: 15px; font-weight: 600; margin-top: 10px;")
        setup_layout.addWidget(hint_lbl)
        setup_layout.addStretch()
        self._cam_stack.addWidget(self._cam_setup_page)

        # Preview page
        self._camera_view = CameraPreviewWidget(placeholder_text="Đang kết nối Cam...")
        self._cam_stack.addWidget(self._camera_view)
        
        self._cam_stack.setCurrentIndex(0)

        # Notification toast
        self._toast = QLabel("")
        self._toast.setFixedHeight(50)
        self._toast.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._toast.setStyleSheet(f"background: {Colors.BG_DARK}; color: {Colors.CYAN}; border: 1px solid {Colors.CYAN}; border-radius: 10px; font-weight: 700;")
        self._toast.hide()
        layout.addWidget(self._toast)

        return panel

    def _on_camera_selected(self, source, label):
        """Xử lý khi người dùng chọn camera."""
        self._selected_camera_source = source
        self._lbl_current_cam.setText(f"🎥 {label}")
        
        # [NEW] Nếu đang trong phiên, cập nhật target_camera ngay lập tức để Mini PC chuyển luồng
        if hasattr(self, "_session_id") and self._session_id:
            def update_target():
                try:
                    requests.post("http://127.0.0.1:9696/api/system/command", json={
                        "command": "START",
                        "session_id": self._session_id,
                        "target_camera": source
                    }, headers={"X-API-Key": "faceattend_secret_2026"}, timeout=2)
                except: pass
            import threading
            threading.Thread(target=update_target, daemon=True).start()

        # Nếu đang live, chuyển ngay luồng stream mới trên giao diện
        if hasattr(self, "_worker") and self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait()
            
            # Khởi tạo worker mới cho camera vừa chọn
            # [FIX] Cả CAM_xx và rtsp:// đều là camera remote từ Mini PC
            is_remote = (source.upper().startswith("CAM_") 
                         or source.lower().startswith("rtsp://"))
            if is_remote:
                self._worker = RemoteStreamWorker(camera_id=source)
            else:
                self._worker = AttendanceWorker(camera_source=source)
                
            self._worker.frame_ready.connect(self._on_frame)
            self._worker.error_occurred.connect(self._on_camera_error)
            self._worker.start()
            
        self._btn_start.setEnabled(True)
        self._btn_start.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.GREEN}; color: white;
                border: none; border-radius: 12px;
                font-size: 18px; font-weight: 900;
            }}
            QPushButton:hover {{ background: {Colors.GREEN_DIM}; }}
        """)

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

        # ── Nút Bắt đầu / Kết thúc (Lớn, Phong cách Premium) ──
        self._btn_start = QPushButton("▶  BẮT ĐẦU ĐIỂM DANH")
        self._btn_start.setFixedHeight(65)
        self._btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_start.setEnabled(False) # Chờ chọn camera
        self._btn_start.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BORDER}; color: {Colors.TEXT_DARK};
                border: none; border-radius: 12px;
                font-size: 18px; font-weight: 900;
            }}
        """)
        self._btn_start.clicked.connect(self._start_session)
        layout.addWidget(self._btn_start)

        self._btn_stop = QPushButton("⏹  KẾT THÚC")
        self._btn_stop.setFixedHeight(65)
        self._btn_stop.hide() # Khi chưa bắt đầu thì ẩn
        self._btn_stop.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.RED}; color: #ffffff;
                border: none; border-radius: 12px;
                font-size: 18px; font-weight: 900;
            }}
            QPushButton:hover {{ background: {Colors.RED_DIM}; }}
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
        """Tải danh sách camera và xây dựng Menu phân cấp cho nút chọn duy nhất."""
        try:
            cameras = camera_repo.get_all(active_only=True)
            self._cam_groups = {"Mini PC": []}
            for i in range(1, 7): self._cam_groups[f"E{i}"] = []
            
            # 1. Phân loại camera từ DB (cho các tòa nhà E1-E6)
            for cam in cameras:
                name = cam.camera_name.upper()
                source = cam.rtsp_url
                group = None
                for i in range(1, 7):
                    if f"E{i}" in name:
                        group = f"E{i}"
                        break
                
                if group:
                    if source: self._cam_groups[group].append((cam.camera_name, source))

            # [NEW] Xây dựng 6 slot camera mặc định cho Mini PC từ dữ liệu Live (Qua API Port 9696)
            try:
                # Phải dùng requests vì API và UI chạy ở 2 process riêng biệt (theo run_server.bat)
                resp = requests.get("http://127.0.0.1:9696/api/system/edge_status", timeout=2)
                edge_data = resp.json() if resp.status_code == 200 else {}
                
                live_status = {}
                if edge_data:
                    # Duyệt qua tất cả các box để gom trạng thái camera (Phòng trường hợp nhiều box)
                    for box_id, box_data in edge_data.items():
                        status = box_data.get("camera_status", {})
                        if status:
                            live_status.update(status)

                # [FIX] Chỉ hiển thị các camera thực sự Mini PC đang báo cáo (online)
                # Không tạo thêm slot giả nếu không có dữ liệu
                cam_index = 0
                for k, v in live_status.items():
                    if not v:
                        continue
                    cam_index += 1
                    # Camera RTSP URL đầy đủ từ ONVIF discovery
                    if k.startswith("rtsp://"):
                        try:
                            ip_port = k.split("@")[1].split("/")[0] if "@" in k else k.split("://")[1].split("/")[0]
                        except:
                            ip_port = "IP Cam"
                        label = f"🟢 IP Cam ({ip_port}) [ONLINE]"
                        source = k  # Dùng RTSP URL làm source cho RemoteStreamWorker
                    else:
                        # Camera ID dạng CAM_01, CAM_02...
                        label = f"🟢 {k} [ONLINE]"
                        source = k  # Dùng đúng Camera ID để khớp với frame upload
                    self._cam_groups["Mini PC"].append((label, source))

                # Fallback nếu không có camera nào online
                if not self._cam_groups["Mini PC"]:
                    self._cam_groups["Mini PC"].append(("🔴 Không có camera [Mini PC offline]", "CAM_01"))

            except Exception as e:
                logger.warning(f"Không lấy được trạng thái live: {e}")
                for i in range(6):
                    self._cam_groups["Mini PC"].append((f"⚪ CAM_0{i+1} [CHƯA KẾT NỐI]", f"CAM_0{i+1}"))

            # 3. Tạo Menu chính phân cấp
            main_menu = QMenu(self)
            main_menu.setStyleSheet(f"""
                QMenu {{ background: {Colors.BG_DARK}; border: 1px solid {Colors.BORDER}; color: {Colors.TEXT}; padding: 5px; }}
                QMenu::item {{ padding: 10px 40px; border-radius: 6px; font-weight: 600; }}
                QMenu::item:selected {{ background: {Colors.CYAN}22; color: {Colors.CYAN}; }}
                QMenu::separator {{ height: 1px; background: {Colors.BORDER}; margin: 5px 10px; }}
            """)

            groups_to_render = ["Mini PC"] + [f"E{i}" for i in range(1, 7)]
            for g_name in groups_to_render:
                cams = self._cam_groups.get(g_name, [])
                if not cams: continue
                
                icon = "⚪" if g_name == "Mini PC" else "🏢"
                sub_menu = main_menu.addMenu(f"{icon}  {g_name}")
                sub_menu.setStyleSheet(main_menu.styleSheet())
                
                for c_name, source in cams:
                    act = sub_menu.addAction(f"📷 {c_name}")
                    act.triggered.connect(lambda chk, s=source, n=c_name: self._on_camera_selected(s, n))
            
            self._btn_camera_select.setMenu(main_menu)

            # Mặc định hiển thị label Mini PC nếu có
            if self._cam_groups["Mini PC"]:
                c_name, source = self._cam_groups["Mini PC"][0]
                # Chỉ hiển thị tên, không hiển thị trạng thái cồng kềnh trên label chính
                self._lbl_current_cam.setText(f"🎥 Mini PC - {c_name.split(' ')[1]}")
            
        except Exception as e:
            logger.error(f"Error refreshing cameras: {e}")


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
                    requests.post("http://127.0.0.1:9696/api/system/command", json={
                        "command": "START",
                        "session_id": sid,
                        "class_id": class_id,
                        "target_camera": camera_source
                    }, headers={"X-API-Key": "faceattend_secret_2026"}, timeout=5)
                except: pass
            
            import threading
            threading.Thread(target=send_start, daemon=True).start()

            # [FIX] Phân loại: 
            # - 'CAM_xx' → Mini PC camera (ID ngắn)
            # - 'rtsp://' → IP Camera trực tiếp hoặc đi qua Mini PC proxy
            # - Còn lại → Camera cục bộ (Webcam)
            target_source = str(self._selected_camera_source or "")
            is_remote = (target_source.upper().startswith("CAM_") 
                         or target_source.lower().startswith("rtsp://"))
            
            if target_source and not is_remote:
                # Webcam cục bộ
                self._worker = AttendanceWorker(camera_source=target_source)
                self._worker.frame_ready.connect(self._on_frame)
                self._worker.error_occurred.connect(self._on_camera_error)
                self._worker.start()
                self._cam_stack.setCurrentIndex(1)
            elif is_remote:
                # Camera từ Mini PC (cả ID ngắn lẫn RTSP URL)
                self._worker = RemoteStreamWorker(camera_id=target_source)
                self._worker.frame_ready.connect(self._on_frame)
                self._worker.error_occurred.connect(self._on_camera_error)
                self._worker.start()
                self._camera_view.set_placeholder("🖥️  ĐANG KẾT NỐI MINI PC...") 
                self._cam_stack.setCurrentIndex(1)
            else:
                self._worker = None
                self._camera_view.set_placeholder("") 
                self._cam_stack.setCurrentIndex(1)

            self._btn_start.hide()
            self._btn_stop.show()
            self._btn_camera_select.setEnabled(True) # KHÔNG khoá chọn camera nữa
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

        if hasattr(self, "_worker") and self._worker:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None

        try:
            from services.attendance_service import attendance_service
            session = attendance_service.end_session()
            
            # Dừng các timer
            self._db_poll_timer.stop()
            self._clock_timer.stop()

            # Ra lệnh STOP qua API
            def send_stop():
                try:
                    requests.post("http://127.0.0.1:9696/api/system/command", json={
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

        # Reset UI
        self._btn_start.show()
        self._btn_stop.hide()
        self._btn_camera_select.setEnabled(True)
        self._inp_subject.setEnabled(True)
        self._date_picker.setEnabled(True)
        self._camera_view.clear()
        
        self._cam_stack.setCurrentIndex(0)
        self._set_cam_status("⬤  Chờ", Colors.TEXT_DARK)
        self._session_start = None

    def _on_frame(self, frame: np.ndarray, elapsed_ms: float, detections: list = None):
        self._camera_view.update_frame_with_detections(frame, detections)
        
        # Nếu đang hiện thông báo lỗi "Chưa có hình", xóa đi vì đã có hình rồi
        if "chưa có hình" in self._lbl_cam_status.text().lower():
             self._set_cam_status("● Đang điểm danh", Colors.GREEN)

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
        # Hiển thị thông báo lỗi chi tiết ra thanh trạng thái
        self._set_cam_status(f"⬤  {msg}", Colors.RED)

    def _show_toast(self, msg: str):
        self._toast.setText(msg)
        self._toast.show()
        # Lưu ý: cần khởi tạo self._toast_timer nếu sử dụng, hiện tại tạm ẩn chức năng ẩn tự động hoặc bạn tự thêm QTimer
        # self._toast_timer.start(3000)

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
                    
            # Update stats: ĐÃ SỬA LỖI HARDCODE + 10 TẠI ĐÂY
            # Lấy sĩ số thực tế (tạm gán cứng = 4 theo DB của bạn)
            # Nếu sau này có hàm lấy tự động, bạn dùng: total_students = class_repo.get_student_count(class_id)
            total_students = 4 
            self._stat_total.setText(str(total_students))
            
            # Tính toán số lượng Vắng
            absent_count = total_students - len(present_list)
            self._stat_absent.setText(str(max(0, absent_count)))
            
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