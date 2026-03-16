"""
ui/pages/settings_page.py
Trang cấu hình hệ thống — Cho phép tùy chỉnh Camera, AI và Database.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QLineEdit, QCheckBox, QComboBox, QGroupBox, QFormLayout,
    QScrollArea, QFrame, QMessageBox, QDoubleSpinBox, QSpinBox
)
from PyQt6.QtCore import Qt
from loguru import logger
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ui.styles.theme import Colors, card_style
from config import db_config, ai_config, camera_config, app_config

class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._load_current_settings()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(30, 30, 30, 30)
        root.setSpacing(20)

        # --- Header ---
        header = QVBoxLayout()
        title = QLabel("Cấu Hình Hệ Thống")
        title.setStyleSheet(f"font-size: 24px; font-weight: 800; color: {Colors.TEXT};")
        subtitle = QLabel("Tinh chỉnh thông số vận hành AI, Camera và Kết nối dữ liệu")
        subtitle.setStyleSheet(f"font-size: 13px; color: {Colors.TEXT_DIM};")
        header.addWidget(title)
        header.addWidget(subtitle)
        root.addLayout(header)

        # --- Scroll Area for settings groups ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        self.container_layout = QVBoxLayout(container)
        self.container_layout.setSpacing(25)

        # 1. Nhóm Camera
        self._group_camera = self._create_group("📹  CẤU HÌNH CAMERA")
        cam_form = QFormLayout(self._group_camera)
        self._set_form_style(cam_form)
        
        self.edit_cam_source = QLineEdit()
        self.edit_cam_source.setPlaceholderText("0 hoặc RTSP URL")
        self.spin_cam_fps = QSpinBox()
        self.spin_cam_fps.setRange(1, 60)
        self.spin_frame_skip = QSpinBox()
        self.spin_frame_skip.setRange(1, 30)
        self.spin_frame_skip.setToolTip("Xử lý 1 frame sau mỗi N frames (Tăng hiệu suất)")

        cam_form.addRow("Nguồn Camera:", self.edit_cam_source)
        cam_form.addRow("FPS Thiết lập:", self.spin_cam_fps)
        cam_form.addRow("Tần suất xử lý (N):", self.spin_frame_skip)

        # 2. Nhóm AI Engine
        self._group_ai = self._create_group("🧠  CẤU HÌNH NHẬN DẠNG (AI)")
        ai_form = QFormLayout(self._group_ai)
        self._set_form_style(ai_form)

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.1, 0.9)
        self.spin_threshold.setSingleStep(0.05)
        self.cmb_det_size = QComboBox()
        self.cmb_det_size.addItems(["320x320 (Nhanh)", "640x640 (Chuẩn)"])
        self.check_gpu = QCheckBox("Sử dụng NVIDIA GPU (CUDA)")

        ai_form.addRow("Ngưỡng nhận diện:", self.spin_threshold)
        ai_form.addRow("Kích thước quét:", self.cmb_det_size)
        ai_form.addRow("Tăng tốc phần cứng:", self.check_gpu)

        # 3. Nhóm Database
        self._group_db = self._create_group("🗄️  CƠ SỞ DỮ LIỆU (SQL SERVER)")
        db_form = QFormLayout(self._group_db)
        self._set_form_style(db_form)

        self.edit_db_server = QLineEdit()
        self.edit_db_name = QLineEdit()
        self.check_win_auth = QCheckBox("Sử dụng Windows Authentication")

        db_form.addRow("Địa chỉ Server:", self.edit_db_server)
        db_form.addRow("Tên Database:", self.edit_db_name)
        db_form.addRow("Xác thực:", self.check_win_auth)

        # Add groups to container
        self.container_layout.addWidget(self._group_camera)
        self.container_layout.addWidget(self._group_ai)
        self.container_layout.addWidget(self._group_db)
        self.container_layout.addStretch()

        scroll.setWidget(container)
        root.addWidget(scroll)

        # --- Action Buttons ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.btn_save = QPushButton("💾  Lưu cấu hình")
        self.btn_save.setFixedSize(160, 45)
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.CYAN}; color: white;
                border-radius: 8px; font-weight: 800; font-size: 14px;
            }}
            QPushButton:hover {{ background-color: {Colors.CYAN_DIM}; }}
        """)
        self.btn_save.clicked.connect(self._save_settings)
        
        btn_layout.addWidget(self.btn_save)
        root.addLayout(btn_layout)

    def _create_group(self, title: str) -> QGroupBox:
        group = QGroupBox(title)
        group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: 800; font-size: 11px; color: {Colors.CYAN};
                border: 1px solid {Colors.BORDER}; border-radius: 12px;
                margin-top: 20px; background: {Colors.BG_CARD}; padding-top: 15px;
            }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 15px; padding: 0 5px; }}
        """)
        return group

    def _set_form_style(self, form: QFormLayout):
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setContentsMargins(20, 20, 20, 20)
        form.setSpacing(15)

    def _load_current_settings(self):
        # Load từ config.py (vốn đã load từ .env hoặc mặc định)
        self.edit_cam_source.setText(str(camera_config.source))
        self.spin_cam_fps.setValue(camera_config.fps)
        self.spin_frame_skip.setValue(camera_config.process_every_n_frames)

        self.spin_threshold.setValue(ai_config.recognition_threshold)
        self.cmb_det_size.setCurrentIndex(1 if ai_config.det_size[0] == 640 else 0)
        self.check_gpu.setChecked(ai_config.gpu_ctx_id >= 0)

        self.edit_db_server.setText(db_config.server)
        self.edit_db_name.setText(db_config.database)
        self.check_win_auth.setChecked(db_config.use_windows_auth)

    def _save_settings(self):
        # Trong thực tế, bạn nên ghi các giá trị này vào file .env
        # Ở đây tôi hướng dẫn cập nhật nóng vào bộ nhớ ứng dụng
        try:
            camera_config.source = self.edit_cam_source.text()
            camera_config.process_every_n_frames = self.spin_frame_skip.value()
            ai_config.recognition_threshold = self.spin_threshold.value()
            db_config.server = self.edit_db_server.text()
            
            # TODO: Triển khai hàm ghi file .env để lưu vĩnh viễn
            
            QMessageBox.information(self, "Thành công", "Đã cập nhật cấu hình.\nMột số thay đổi yêu cầu khởi động lại ứng dụng.")
            logger.info("Người dùng đã cập nhật cài đặt hệ thống.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể lưu: {str(e)}")