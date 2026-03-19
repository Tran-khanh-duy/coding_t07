"""
ui/pages/enroll_page.py
"""
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QComboBox,
    QGroupBox, QScrollArea, QFrame, QSizePolicy,
    QMessageBox, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter,
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize,
)
from PyQt6.QtGui import QFont, QPixmap, QImage, QColor
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.styles.theme import Colors, card_style, badge_style, combo_style, input_style
from ui.widgets.camera_preview import CameraPreviewWidget
from config import CAMERAS


# ─────────────────────────────────────────────
#  Worker: chụp ảnh từ camera trong thread riêng
# ─────────────────────────────────────────────
class CaptureWorker(QThread):
    """Thread chụp ảnh liên tục từ webcam/IP camera."""
    frame_ready   = pyqtSignal(np.ndarray)   # Frame mới để hiển thị
    photo_taken   = pyqtSignal(int, int)     # (current, total)
    capture_done  = pyqtSignal(list)         # Danh sách frames đã chụp
    face_detected = pyqtSignal(bool)         # Có phát hiện khuôn mặt không

    def __init__(self, source=0, target_count=10, parent=None):
        super().__init__(parent)
        self.source       = source
        self.target_count = target_count
        self._capturing   = False
        self._running     = True
        self._frames      = []
        self._last_capture_time = 0

    def start_capture(self):
        self._capturing = True
        self._frames    = []

    def stop_capture(self):
        self._capturing = False

    def stop(self):
        self._running   = False
        self._capturing = False

    def run(self):
        import time
        from config import camera_config
        
        # Đổi str thành int nếu là webcam USB
        source = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            logger.error(f"Không mở được camera source={self.source}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  camera_config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.height)
        cap.set(cv2.CAP_PROP_FPS, camera_config.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        from services.face_engine import face_engine
        
        # Tách AI sang thread khác để không block camera
        import threading
        class UIDetectThread(threading.Thread):
            def __init__(self, worker):
                super().__init__(daemon=True)
                self.worker = worker
                self.lock = threading.Lock()
                self.latest_frame = None
                self._running = True
                self.latest_faces = []
                self.has_face = False
                
            def update_frame(self, frame):
                with self.lock:
                    self.latest_frame = frame.copy()
            
            def get_latest(self):
                with self.lock:
                    return self.has_face, self.latest_faces
                    
            def stop(self):
                self._running = False
                
            def run(self):
                while self._running:
                    with self.lock:
                        frame = self.latest_frame
                        self.latest_frame = None
                        
                    if frame is None:
                        time.sleep(0.01)
                        continue
                        
                    has_face = False
                    faces = []
                    if face_engine.is_ready:
                        try:
                            detected_faces = face_engine.detect_faces(frame)
                            if detected_faces:
                                # Chọn khuôn mặt lớn nhất (gần camera nhất) thay vì yêu cầu chỉ có 1 mặt
                                main_face = max(detected_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                                faces = [main_face]
                                has_face = True
                        except Exception:
                            pass
                    
                    with self.lock:
                        self.latest_faces = faces
                        self.has_face = has_face
                        
                    self.worker.face_detected.emit(has_face)

        ai_thread = UIDetectThread(self)
        ai_thread.start()
        
        frame_skip = 0

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_skip += 1
            if frame_skip >= camera_config.process_every_n_frames:
                frame_skip = 0
                ai_thread.update_frame(frame)
                
            has_face, faces = ai_thread.get_latest()

            # Lưu frame GỐC cho embedding
            if self._capturing and has_face:
                now = time.time()
                if now - self._last_capture_time >= 0.4:
                    self._frames.append(frame.copy())
                    self._last_capture_time = now
                    count = len(self._frames)
                    self.photo_taken.emit(count, self.target_count)
                    if count >= self.target_count:
                        self._capturing = False
                        self.capture_done.emit(self._frames.copy())

            # Flip ngang để hiển thị như gương — chỉ dùng cho UI
            display = cv2.flip(frame, 1)
            w = display.shape[1]

            # Vẽ bounding box trên frame đã flip
            if face_engine.is_ready and has_face:
                try:
                    for f in faces:
                        x1, y1, x2, y2 = f.bbox
                        x1m, x2m = w - x2, w - x1
                        color = (0, 220, 80)
                        cv2.rectangle(display, (int(x1m), int(y1)), (int(x2m), int(y2)), color, 2)
                        cv2.putText(display, f"{f.det_score:.2f}",
                                    (int(x1m), int(y1) - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except Exception:
                    pass

            # Vẽ khung guide căn giữa
            h, w2 = display.shape[:2]
            cx, cy = w2 // 2, h // 2
            gw, gh = 260, 320
            cv2.rectangle(display,
                          (cx - gw // 2, cy - gh // 2),
                          (cx + gw // 2, cy + gh // 2),
                          (80, 80, 80), 1)

            self.frame_ready.emit(display)

        ai_thread.stop()
        ai_thread.join(timeout=1.0)
        cap.release()


# ─────────────────────────────────────────────
#  EnrollWorker — chạy finish_enrollment() trên thread riêng
# ─────────────────────────────────────────────
class EnrollWorker(QThread):
    """Thread xử lý trích xuất embedding + lưu DB để không block UI."""
    done = pyqtSignal(object)  # kết quả (EnrollResult)

    def __init__(self, frames: list, student_id: int, parent=None):
        super().__init__(parent)
        self._frames = frames
        self._student_id = student_id

    def run(self):
        try:
            from services.enrollment_service import enrollment_service
            enrollment_service.start_capture(
                student_id=self._student_id,
                photo_count=len(self._frames)
            )
            enrollment_service._capture.frames = self._frames.copy()
            result = enrollment_service.finish_enrollment()
            self.done.emit(result)
        except Exception as e:
            logger.error(f"EnrollWorker error: {e}")
            # Phát một kết quả lỗi giả
            class _Err:
                success = False
                error_msg = str(e)
            self.done.emit(_Err())


# ─────────────────────────────────────────────
#  EnrollPage
# ─────────────────────────────────────────────
class EnrollPage(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._capture_worker: CaptureWorker | None = None
        self._enroll_worker: EnrollWorker | None = None
        self._captured_frames: list = []
        self._current_student_id: int | None = None
        self._camera_active = False
        self._mode = "create"   # "create" hoặc "update"

        self._setup_ui()
        self._load_classes()

    # ─── UI ───────────────────────────────────

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: transparent;")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        main = QVBoxLayout(container)
        main.setContentsMargins(28, 24, 28, 24)
        main.setSpacing(20)

        # ── Header ──
        header = QHBoxLayout()
        title = QLabel("Đăng Ký Học Viên")
        title.setStyleSheet(f"font-size: 24px; font-weight: 900; color: {Colors.TEXT};")
        subtitle = QLabel("Nhập thông tin và chụp 10 ảnh khuôn mặt")
        subtitle.setStyleSheet(f"font-size: 14px; color: {Colors.TEXT_DIM};")
        self._title_lbl    = title
        self._subtitle_lbl = subtitle
        
        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        header.addLayout(title_col)
        header.addStretch()

        # Nút reset form
        self._btn_reset = QPushButton("🔄  Form Mới")
        self._btn_reset.setMinimumHeight(44)
        self._btn_reset.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_CARD};
                color: {Colors.TEXT_DIM};
                border: 1px solid {Colors.BORDER_LT};
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{ color: {Colors.TEXT}; background: {Colors.BG_HOVER}; border-color: {Colors.CYAN}; }}
        """)
        self._btn_reset.clicked.connect(self._reset_form)
        header.addWidget(self._btn_reset)
        main.addLayout(header)

        # ── Content: Left + Right ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background: transparent; width: 12px; }")

        # LEFT: Form thông tin
        left = self._build_form_panel()
        splitter.addWidget(left)

        # RIGHT: Camera + Progress
        right = self._build_camera_panel()
        splitter.addWidget(right)

        splitter.setSizes([450, 650]) # Tăng kích thước để chống cắt chữ
        main.addWidget(splitter, 1)

        scroll.setWidget(container)
        outer.addWidget(scroll)

    def _build_form_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(card_style(Colors.BORDER, radius=12))
        layout = QVBoxLayout(panel)
        layout.setSpacing(18)
        layout.setContentsMargins(24, 24, 24, 24)

        # Tiêu đề
        sec_title = QLabel("THÔNG TIN HỌC VIÊN")
        sec_title.setStyleSheet(
            f"font-size: 12px; font-weight: 800; color: {Colors.TEXT_DIM}; letter-spacing: 1.5px; border: none; background: transparent;"
        )
        layout.addWidget(sec_title)

        # Form fields
        def make_field(label: str, placeholder: str = "") -> tuple[QLabel, QLineEdit]:
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 13px; font-weight: 600; border: none; background: transparent;")
            inp = QLineEdit()
            inp.setPlaceholderText(placeholder)
            inp.setStyleSheet(input_style())
            inp.setMinimumHeight(42)
            return lbl, inp

        lbl, self._inp_code = make_field("Mã học viên *", "VD: HV001")
        layout.addWidget(lbl)
        layout.addWidget(self._inp_code)

        lbl, self._inp_name = make_field("Họ và tên *", "VD: Nguyễn Văn A")
        layout.addWidget(lbl)
        layout.addWidget(self._inp_name)

        # Lớp học
        cls_lbl = QLabel("Lớp học")
        cls_lbl.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 13px; font-weight: 600; border: none; background: transparent;")
        self._cmb_class = QComboBox()
        self._cmb_class.setStyleSheet(combo_style())
        self._cmb_class.setMinimumHeight(42)
        layout.addWidget(cls_lbl)
        layout.addWidget(self._cmb_class)

        # Giới tính
        gt_lbl = QLabel("Giới tính")
        gt_lbl.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 13px; font-weight: 600; border: none; background: transparent;")
        self._cmb_gender = QComboBox()
        self._cmb_gender.addItems(["-- Chọn --", "Nam", "Nữ", "Khác"])
        self._cmb_gender.setStyleSheet(combo_style())
        self._cmb_gender.setMinimumHeight(42)
        layout.addWidget(gt_lbl)
        layout.addWidget(self._cmb_gender)

        # Số điện thoại & Email
        lbl, self._inp_phone = make_field("Số điện thoại", "VD: 0901234567")
        layout.addWidget(lbl)
        layout.addWidget(self._inp_phone)

        lbl, self._inp_email = make_field("Email", "VD: hocvien@email.com")
        layout.addWidget(lbl)
        layout.addWidget(self._inp_email)

        # Chọn Camera
        cam_lbl = QLabel("Chọn Camera chụp ảnh")
        cam_lbl.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 13px; font-weight: 600; border: none; background: transparent;")
        self._cmb_camera = QComboBox()
        self._cmb_camera.setStyleSheet(combo_style())
        self._cmb_camera.setMinimumHeight(42)
        for cam in CAMERAS:
            self._cmb_camera.addItem(cam['name'], cam['source'])
        layout.addWidget(cam_lbl)
        layout.addWidget(self._cmb_camera)

        layout.addStretch()

        # Nút tạo học viên
        self._btn_create = QPushButton("✅  TẠO HỌC VIÊN")
        self._btn_create.setFixedHeight(50)
        self._btn_create.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.CYAN};
                color: #ffffff;
                border: none;
                border-radius: 10px;
                font-size: 15px;
                font-weight: 800;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{ background: {Colors.CYAN_DIM}; }}
            QPushButton:disabled {{
                background: {Colors.BORDER};
                color: {Colors.TEXT_DARK};
            }}
        """)
        self._btn_create.clicked.connect(self._on_create_student)
        layout.addWidget(self._btn_create)

        # Trạng thái tạo HV
        self._lbl_create_status = QLabel("")
        self._lbl_create_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_create_status.setWordWrap(True)
        self._lbl_create_status.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {Colors.TEXT_DIM}; border: none; background: transparent;")
        layout.addWidget(self._lbl_create_status)

        return panel

    def _build_camera_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(panel)
        layout.setSpacing(18)
        layout.setContentsMargins(0, 0, 0, 0)

        # Camera preview
        self._camera_view = CameraPreviewWidget(
            placeholder_text="📷  Nhấn 'Mở Camera' để bắt đầu"
        )
        self._camera_view.setMinimumHeight(400)
        self._camera_view.setStyleSheet(f"background: {Colors.CAM_BG}; border-radius: 12px;")
        layout.addWidget(self._camera_view, 1)

        # Guide label
        self._lbl_guide = QLabel("💡  Đứng thẳng, nhìn vào camera, đảm bảo đủ ánh sáng")
        self._lbl_guide.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_guide.setStyleSheet(f"""
            background: {Colors.CYAN}18;
            color: {Colors.CYAN};
            border: 1px solid {Colors.CYAN}44;
            border-radius: 10px;
            padding: 12px 18px;
            font-size: 14px;
            font-weight: 700;
        """)
        layout.addWidget(self._lbl_guide)

        # Progress chụp ảnh
        prog_card = QWidget()
        prog_card.setStyleSheet(card_style(Colors.BORDER, radius=12))
        prog_layout = QVBoxLayout(prog_card)
        prog_layout.setSpacing(12)
        prog_layout.setContentsMargins(20, 18, 20, 18)

        prog_header = QHBoxLayout()
        prog_title = QLabel("TIẾN TRÌNH CHỤP ẢNH")
        prog_title.setStyleSheet(
            f"font-size: 12px; font-weight: 800; color: {Colors.TEXT_DIM}; letter-spacing: 1.5px; border: none; background: transparent;"
        )
        self._lbl_count = QLabel("0 / 10")
        self._lbl_count.setStyleSheet(
            f"font-size: 16px; font-weight: 900; color: {Colors.CYAN}; border: none; background: transparent;"
        )
        prog_header.addWidget(prog_title)
        prog_header.addStretch()
        prog_header.addWidget(self._lbl_count)
        prog_layout.addLayout(prog_header)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 10)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(14)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {Colors.BG_INPUT};
                border: 1px solid {Colors.BORDER_LT};
                border-radius: 7px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {Colors.CYAN_DIM}, stop:1 {Colors.CYAN});
                border-radius: 7px;
            }}
        """)
        prog_layout.addWidget(self._progress_bar)

        # Dots hiển thị từng ảnh
        self._dots_layout = QHBoxLayout()
        self._dots_layout.setSpacing(8)
        self._dots: list[QLabel] = []
        for i in range(10):
            dot = QLabel("○")
            dot.setFixedWidth(24)
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dot.setStyleSheet(f"color: {Colors.BORDER_LT}; font-size: 18px; border: none; background: transparent;")
            self._dots.append(dot)
            self._dots_layout.addWidget(dot)
        self._dots_layout.addStretch()
        
        # Face detection indicator
        self._lbl_face_status = QLabel("⬤  Chưa phát hiện khuôn mặt")
        self._lbl_face_status.setStyleSheet(
            f"color: {Colors.TEXT_DARK}; font-size: 13px; font-weight: 700; border: none; background: transparent;"
        )
        self._dots_layout.addWidget(self._lbl_face_status)
        prog_layout.addLayout(self._dots_layout)

        layout.addWidget(prog_card)

        # Nút camera actions
        btn_row = QHBoxLayout()
        btn_row.setSpacing(14)

        self._btn_camera = QPushButton("📷  Mở Camera")
        self._btn_camera.setFixedHeight(50)
        self._btn_camera.setEnabled(False)
        self._btn_camera.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_CARD};
                color: {Colors.TEXT};
                border: 1px solid {Colors.BORDER_LT};
                border-radius: 10px;
                font-size: 15px;
                font-weight: 700;
            }}
            QPushButton:hover {{ background: {Colors.BG_HOVER}; border-color: {Colors.CYAN}; }}
        """)
        self._btn_camera.clicked.connect(self._toggle_camera)

        self._btn_capture = QPushButton("📸  Bắt Đầu Chụp")
        self._btn_capture.setFixedHeight(50)
        self._btn_capture.setEnabled(False)
        self._btn_capture.setStyleSheet(f"""
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
        self._btn_capture.clicked.connect(self._start_capture)

        self._btn_enroll = QPushButton("🎯  Hoàn Tất Đăng Ký")
        self._btn_enroll.setFixedHeight(50)
        self._btn_enroll.setEnabled(False)
        self._btn_enroll.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ORANGE};
                color: #ffffff;
                border: none;
                border-radius: 10px;
                font-size: 15px;
                font-weight: 800;
            }}
            QPushButton:hover {{ background: #D97706; }}
            QPushButton:disabled {{
                background: {Colors.BORDER};
                color: {Colors.TEXT_DARK};
            }}
        """)
        self._btn_enroll.clicked.connect(self._finish_enrollment)

        btn_row.addWidget(self._btn_camera)
        btn_row.addWidget(self._btn_capture)
        btn_row.addWidget(self._btn_enroll)
        layout.addLayout(btn_row)

        # Kết quả enrollment
        self._result_card = QWidget()
        self._result_card.setStyleSheet(card_style(Colors.BORDER, radius=12))
        self._result_card.hide()
        res_layout = QVBoxLayout(self._result_card)
        res_layout.setContentsMargins(16, 16, 16, 16)
        self._lbl_result = QLabel()
        self._lbl_result.setWordWrap(True)
        self._lbl_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_result.setStyleSheet(f"font-size: 15px; font-weight: 800; border: none; background: transparent;")
        res_layout.addWidget(self._lbl_result)
        layout.addWidget(self._result_card)

        return panel

    # ─── Public: Load học viên từ trang Danh sách ──

    def load_student(self, student_id: int):
        try:
            from database.repositories import student_repo
            student = student_repo.get_by_id(student_id)
            if student is None:
                logger.error(f"Không tìm thấy học viên id={student_id}")
                return
        except Exception as e:
            logger.error(f"load_student error: {e}")
            return

        self._reset_form()
        self._mode = "update" if student.face_enrolled else "create"
        self._current_student_id = student_id

        self._inp_code.setText(student.student_code or "")
        self._inp_name.setText(student.full_name or "")
        self._inp_phone.setText(student.phone or "")
        self._inp_email.setText(student.email or "")

        gender_map = {"Nam": 1, "Nữ": 2, "Khác": 3}
        idx = gender_map.get(student.gender, 0)
        self._cmb_gender.setCurrentIndex(idx)

        for i in range(self._cmb_class.count()):
            if self._cmb_class.itemData(i) == student.class_id:
                self._cmb_class.setCurrentIndex(i)
                break

        if self._mode == "update":
            self._setup_update_mode(student)
        else:
            self._setup_create_mode_prefilled(student)

    def _setup_update_mode(self, student):
        self._title_lbl.setText("Cập Nhật Học Viên")
        self._subtitle_lbl.setText(f"Chỉnh sửa thông tin và/hoặc chụp lại ảnh — [{student.student_code}]")
        
        for w in [self._inp_code, self._inp_name, self._cmb_class, self._cmb_gender, self._inp_phone, self._inp_email]:
            w.setEnabled(True)

        self._btn_create.setText("💾  LƯU CẬP NHẬT")
        self._btn_create.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ORANGE}; color: #ffffff;
                border: none; border-radius: 10px; font-size: 15px; font-weight: 800; letter-spacing: 1px;
            }}
            QPushButton:hover {{ background: #D97706; }}
            QPushButton:disabled {{ background: {Colors.BORDER}; color: {Colors.TEXT_DARK}; }}
        """)
        self._btn_create.setEnabled(True)
        try: self._btn_create.clicked.disconnect()
        except: pass
        self._btn_create.clicked.connect(self._on_update_student)
        self._btn_camera.setEnabled(True)
        self._set_create_status(f"✏️  Đang chỉnh sửa: [{student.student_code}] {student.full_name}", Colors.ORANGE)

    def _setup_create_mode_prefilled(self, student):
        self._title_lbl.setText("Đăng Ký Học Viên")
        self._subtitle_lbl.setText(f"Thông tin đã điền — [{student.student_code}] chỉ cần mở camera và chụp")
        self._lock_form()
        self._btn_create.setEnabled(False)
        self._btn_camera.setEnabled(True)
        self._set_create_status(f"✅ Học viên: [{student.student_code}] {student.full_name} — Hãy chụp ảnh", Colors.GREEN)

    def _on_update_student(self):
        code = self._inp_code.text().strip()
        name = self._inp_name.text().strip()
        if not code or not name:
            self._set_create_status("⚠️ Vui lòng nhập đầy đủ mã và tên!", Colors.ORANGE)
            return

        class_id = self._cmb_class.currentData()
        gender   = self._cmb_gender.currentText()
        gender   = None if gender == "-- Chọn --" else gender
        phone    = self._inp_phone.text().strip() or None
        email    = self._inp_email.text().strip() or None

        try:
            from database.repositories import student_repo
            updated = student_repo.update(
                student_id=self._current_student_id, student_code=code, full_name=name,
                class_id=class_id, gender=gender, phone=phone, email=email,
            )
            if not updated:
                self._set_create_status("❌ Lỗi cập nhật thông tin!", Colors.RED)
                return

            if self._captured_frames:
                self._btn_create.setEnabled(False)
                self._btn_create.setText("⏳  Đang xử lý ảnh...")
                
                self._enroll_worker = EnrollWorker(
                    frames=self._captured_frames,
                    student_id=self._current_student_id
                )
                self._enroll_worker.done.connect(self._on_update_enroll_done)
                self._enroll_worker.start()
            else:
                self._set_create_status(f"✅ Đã cập nhật thông tin [{code}] {name} (không có ảnh mới)", Colors.GREEN)
                self._btn_create.setText("✅  Đã cập nhật")
                self._btn_create.setEnabled(False)
        except Exception as e:
            logger.error(f"_on_update_student error: {e}")
            self._set_create_status(f"❌ Lỗi: {e}", Colors.RED)
            self._btn_create.setEnabled(True)

    def _on_update_enroll_done(self, result):
        if result.success:
            self._set_create_status(
                f"✅ Cập nhật thành công! Ảnh hợp lệ: {result.photos_valid}/{result.photos_taken}",
                Colors.GREEN
            )
            self._lbl_result.setText(
                f"✅ Đã cập nhật!\n[{result.student_code}] {result.full_name}\nẢnh hợp lệ: {result.photos_valid}/{result.photos_taken}"
            )
            self._lbl_result.setStyleSheet(
                f"font-size: 15px; font-weight: 800; color: {Colors.GREEN}; border: none; background: transparent;"
            )
            self._result_card.setStyleSheet(card_style(Colors.GREEN + "40", radius=12))
            self._result_card.show()
            self._btn_create.setText("✅  Đã cập nhật")
        else:
            self._set_create_status(f"❌ Lỗi xử lý ảnh: {result.error_msg}", Colors.RED)
            self._btn_create.setEnabled(True)
            self._btn_create.setText("💾  LƯU CẬP NHẬT")

    # ─── Logic ────────────────────────────────

    def _load_classes(self):
        self._cmb_class.clear()
        self._cmb_class.addItem("-- Chọn lớp --", None)
        try:
            from database.repositories import class_repo
            for cls in class_repo.get_all():
                self._cmb_class.addItem(f"{cls.class_code} — {cls.class_name}", cls.class_id)
        except Exception as e:
            logger.warning(f"Không load được danh sách lớp: {e}")

    def _on_create_student(self):
        code = self._inp_code.text().strip()
        name = self._inp_name.text().strip()
        if not code or not name:
            self._set_create_status("⚠️ Vui lòng nhập mã và tên!", Colors.ORANGE)
            return

        class_id = self._cmb_class.currentData()
        gender   = self._cmb_gender.currentText()
        gender   = None if gender == "-- Chọn --" else gender
        phone    = self._inp_phone.text().strip() or None
        email    = self._inp_email.text().strip() or None

        try:
            from services.enrollment_service import enrollment_service
            sid = enrollment_service.create_student(
                student_code=code, full_name=name, class_id=class_id, gender=gender, phone=phone, email=email,
            )
            if sid and sid > 0:
                self._current_student_id = sid
                self._set_create_status(f"✅ Đã tạo: [{code}] {name} (ID={sid})", Colors.GREEN)
                self._btn_create.setEnabled(False)
                self._btn_camera.setEnabled(True)
                self._lock_form()
            else:
                self._set_create_status(f"❌ Mã [{code}] đã tồn tại hoặc lỗi DB!", Colors.RED)
        except Exception as e:
            self._set_create_status(f"❌ Lỗi: {e}", Colors.RED)

    def _toggle_camera(self):
        if not self._camera_active: self._open_camera()
        else: self._close_camera()

    def _open_camera(self):
        if self._capture_worker: return
        camera_source = self._cmb_camera.currentData()
        if camera_source is None:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn Camera hợp lệ!")
            return
            
        self._capture_worker = CaptureWorker(source=camera_source, target_count=10)
        self._capture_worker.frame_ready.connect(self._on_frame)
        self._capture_worker.photo_taken.connect(self._on_photo_taken)
        self._capture_worker.capture_done.connect(self._on_capture_done)
        self._capture_worker.face_detected.connect(self._on_face_detected)
        self._capture_worker.start()

        self._camera_active = True
        self._btn_camera.setText("⏹  Đóng Camera")
        self._btn_camera.setStyleSheet(f"""
            QPushButton {{ background: {Colors.RED_DIM}; color: {Colors.TEXT}; border: 1px solid {Colors.RED}44; border-radius: 10px; font-size: 15px; font-weight: 700; }}
            QPushButton:hover {{ background: {Colors.RED}; color: #fff; }}
        """)
        if self._current_student_id: self._btn_capture.setEnabled(True)

    def _close_camera(self):
        if self._capture_worker:
            self._capture_worker.stop()
            self._capture_worker.wait(3000)
            self._capture_worker = None
        self._camera_active = False
        self._camera_view.clear()
        self._camera_view.set_status("")
        self._btn_camera.setText("📷  Mở Camera")
        self._btn_camera.setStyleSheet(f"""
            QPushButton {{ background: {Colors.BG_CARD}; color: {Colors.TEXT}; border: 1px solid {Colors.BORDER_LT}; border-radius: 10px; font-size: 15px; font-weight: 700; }}
            QPushButton:hover {{ background: {Colors.BG_HOVER}; border-color: {Colors.CYAN}; }}
        """)
        self._btn_capture.setEnabled(False)

    def _start_capture(self):
        if not self._current_student_id:
            QMessageBox.warning(self, "Chưa tạo học viên", "Vui lòng tạo học viên trước khi chụp ảnh!")
            return
        if not self._capture_worker: return
        
        self._progress_bar.setValue(0)
        self._lbl_count.setText("0 / 10")
        for dot in self._dots:
            dot.setText("○")
            dot.setStyleSheet(f"color: {Colors.BORDER_LT}; font-size: 18px; border: none; background: transparent;")
            
        self._captured_frames = []
        self._btn_enroll.setEnabled(False)
        self._result_card.hide()
        self._capture_worker.start_capture()
        self._btn_capture.setEnabled(False)
        self._btn_capture.setText("📸  Đang chụp...")
        self._lbl_guide.setText("✅  Nhìn thẳng vào camera — hệ thống đang chụp tự động")

    def _finish_enrollment(self):
        if not self._captured_frames:
            QMessageBox.warning(self, "Chưa có ảnh", "Vui lòng chụp ảnh trước!")
            return
        if not self._current_student_id: return
        if self._enroll_worker and self._enroll_worker.isRunning(): return

        self._btn_enroll.setEnabled(False)
        self._btn_enroll.setText("⏳  Đang xử lý...")
        self._result_card.hide()

        self._enroll_worker = EnrollWorker(
            frames=self._captured_frames,
            student_id=self._current_student_id
        )
        self._enroll_worker.done.connect(self._on_enroll_done)
        self._enroll_worker.start()

    def _on_enroll_done(self, result):
        self._result_card.show()
        if result.success:
            self._lbl_result.setText(
                f"🎉 Đăng ký thành công!\n"
                f"[{result.student_code}] {result.full_name}\n"
                f"Ảnh hợp lệ: {result.photos_valid}/{result.photos_taken} "
                f"| Chất lượng: {result.avg_det_score*100:.1f}%"
            )
            self._lbl_result.setStyleSheet(
                f"font-size: 15px; font-weight: 800; color: {Colors.GREEN}; border: none; background: transparent;"
            )
            self._result_card.setStyleSheet(card_style(Colors.GREEN + "40", radius=12))
            self._btn_enroll.setText("✅  Đã hoàn tất")
        else:
            self._lbl_result.setText(f"❌ {result.error_msg}")
            self._lbl_result.setStyleSheet(
                f"font-size: 15px; font-weight: 800; color: {Colors.RED}; border: none; background: transparent;"
            )
            self._result_card.setStyleSheet(card_style(Colors.RED + "40", radius=12))
            self._btn_enroll.setText("🎯  Thử lại")
            self._btn_enroll.setEnabled(True)

    def _on_frame(self, frame: np.ndarray):
        self._camera_view.update_frame(frame)

    def _on_photo_taken(self, current: int, total: int):
        self._progress_bar.setValue(current)
        self._lbl_count.setText(f"{current} / {total}")
        if current <= len(self._dots):
            self._dots[current - 1].setText("●")
            self._dots[current - 1].setStyleSheet(f"color: {Colors.CYAN}; font-size: 18px; border: none; background: transparent;")

    def _on_capture_done(self, frames: list):
        self._captured_frames = frames
        self._btn_capture.setText("✅  Chụp xong!")
        self._btn_capture.setEnabled(False)
        self._btn_enroll.setEnabled(True)
        self._lbl_guide.setText("✅  Đã chụp xong — Nhấn 'Hoàn Tất Đăng Ký' để lưu")
        for dot in self._dots:
            dot.setText("●")
            dot.setStyleSheet(f"color: {Colors.GREEN}; font-size: 18px; border: none; background: transparent;")

    def _on_face_detected(self, detected: bool):
        if detected:
            self._lbl_face_status.setText("⬤  Phát hiện khuôn mặt ✓")
            self._lbl_face_status.setStyleSheet(f"color: {Colors.GREEN}; font-size: 13px; font-weight: 700; border: none; background: transparent;")
            self._camera_view.set_status("Khuôn mặt OK", Colors.GREEN)
        else:
            self._lbl_face_status.setText("⬤  Chưa phát hiện khuôn mặt")
            self._lbl_face_status.setStyleSheet(f"color: {Colors.TEXT_DARK}; font-size: 13px; font-weight: 700; border: none; background: transparent;")
            self._camera_view.set_status("Đưa mặt vào khung", Colors.ORANGE)

    def _set_create_status(self, msg: str, color: str):
        self._lbl_create_status.setText(msg)
        self._lbl_create_status.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {color}; border: none; background: transparent;")

    def _lock_form(self):
        for w in [self._inp_code, self._inp_name, self._cmb_class, self._cmb_gender, self._inp_phone, self._inp_email, self._cmb_camera]:
            w.setEnabled(False)

    def _reset_form(self):
        self._close_camera()
        self._current_student_id = None
        self._captured_frames    = []
        self._mode               = "create"

        for w in [self._inp_code, self._inp_name, self._cmb_class, self._cmb_gender, self._inp_phone, self._inp_email, self._cmb_camera]:
            w.setEnabled(True)
            if isinstance(w, QLineEdit): w.clear()
            
        self._cmb_class.setCurrentIndex(0)
        self._cmb_gender.setCurrentIndex(0)
        
        self._title_lbl.setText("Đăng Ký Học Viên")
        self._subtitle_lbl.setText("Nhập thông tin và chụp 10 ảnh khuôn mặt")
        self._btn_create.setText("✅  TẠO HỌC VIÊN")
        self._btn_create.setStyleSheet(f"""
            QPushButton {{ background: {Colors.CYAN}; color: #ffffff; border: none; border-radius: 10px; font-size: 15px; font-weight: 800; letter-spacing: 1px; }}
            QPushButton:hover {{ background: {Colors.CYAN_DIM}; }}
            QPushButton:disabled {{ background: {Colors.BORDER}; color: {Colors.TEXT_DARK}; }}
        """)
        self._btn_create.setEnabled(True)
        try: self._btn_create.clicked.disconnect()
        except Exception: pass
        self._btn_create.clicked.connect(self._on_create_student)

        self._btn_camera.setEnabled(False)
        self._btn_capture.setEnabled(False)
        self._btn_capture.setText("📸  Bắt Đầu Chụp")
        self._btn_enroll.setEnabled(False)
        self._btn_enroll.setText("🎯  Hoàn Tất Đăng Ký")
        self._progress_bar.setValue(0)
        self._lbl_count.setText("0 / 10")
        self._result_card.hide()
        self._lbl_create_status.setText("")
        self._lbl_guide.setText("💡  Đứng thẳng, nhìn vào camera, đảm bảo đủ ánh sáng")
        for dot in self._dots:
            dot.setText("○")
            dot.setStyleSheet(f"color: {Colors.BORDER_LT}; font-size: 18px; border: none; background: transparent;")
        self._load_classes()

    def closeEvent(self, event):
        self._close_camera()
        super().closeEvent(event)

    def hideEvent(self, event):
        if self._camera_active: self._close_camera()
        super().hideEvent(event)