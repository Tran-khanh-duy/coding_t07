import subprocess
import platform
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QGridLayout, QScrollArea, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QGraphicsDropShadowEffect, QDateEdit  # <-- Đã thêm QDateEdit ở đây
)
from PyQt6.QtCore import Qt, QTimer, QRectF, pyqtSignal, QSize, QDate  # <-- Đã thêm QDate ở đây
from PyQt6.QtGui import QFont, QColor, QPixmap, QPainter, QPen, QImage, QPainterPath

import sys
import requests
import psutil
from loguru import logger
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.styles.theme import Colors
from config import db_config, ai_config

def get_gpu_name():
    try:
        output = subprocess.check_output("wmic path win32_VideoController get name", shell=True, text=True)
        lines = [line.strip() for line in output.split('\n') if line.strip() and "Name" not in line]
        if lines:
            return lines[0]
    except Exception:
        pass
    return platform.processor()

class CircularProgress(QWidget):
    """Biểu đồ vòng tròn thể hiện tỷ lệ phần trăm."""
    def __init__(self, color_str, bg_color_str, parent=None):
        super().__init__(parent)
        self.setFixedSize(140, 140)
        self.value = 0
        self.max_value = 0 # Đổi mặc định về 0
        self.color = QColor(color_str)
        self.bg_color = QColor(bg_color_str)

    def set_value(self, val, max_val):
        self.value = int(val)
        self.max_value = int(max_val)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = QRectF(15, 15, 110, 110)
        
        # Nền vòng tròn (Vắng mặt)
        pen_bg = QPen(self.bg_color)
        pen_bg.setWidth(14)
        pen_bg.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_bg)
        painter.drawArc(rect, 0, 360 * 16)
        
        # Vòng tròn giá trị (Có mặt) - Chỉ vẽ khi max_value > 0 để tránh chia cho 0
        if self.max_value > 0:
            pen_val = QPen(self.color)
            pen_val.setWidth(14)
            pen_val.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen_val)
            
            span_angle = int((self.value / self.max_value) * 360 * 16)
            painter.drawArc(rect, 90 * 16, -span_angle)
        
        # Chữ ở giữa
        painter.setPen(QColor("#0F172A"))
        font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"Total: {self.max_value}\nStudents")

def create_shadow():
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(15)
    shadow.setColor(QColor(0, 0, 0, 15))
    shadow.setOffset(0, 4)
    return shadow
    
class DashboardPage(QWidget):
    go_to_live_view = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gpu_name = get_gpu_name()
        self.setStyleSheet("background-color: #F1F5F9;")  # Nền xám nhạt nhẹ nhàng
        
        self._students_total = 85
        self._present_count = 72
        self._absent_count = 13

        self._setup_ui()

        self._sys_timer = QTimer(self)
        self._sys_timer.timeout.connect(self._check_system_realtime)
        self._sys_timer.start(3000)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # ==========================================
        # TOP PANEL: LIVE ATTENDANCE LOG (Table)
        # ==========================================
        top_panel = QFrame()
        top_panel.setStyleSheet("""
            QFrame#TopPanel {
                background-color: #FFFFFF;
                border-radius: 12px;
                border: 1px solid #E2E8F0;
            }
        """)
        top_panel.setObjectName("TopPanel")
        top_panel.setGraphicsEffect(create_shadow())
        top_layout = QVBoxLayout(top_panel)
        top_layout.setContentsMargins(20, 20, 20, 10)

        title_top = QLabel("<b>LIVE ATTENDANCE LOG</b> <span style='font-size:13px; font-weight:normal; color:#64748B;'>Hàng điểm danh trực tiếp</span>")
        title_top.setStyleSheet("font-size: 15px; color: #0F172A; border: none; background: transparent;")
        title_top.setTextFormat(Qt.TextFormat.RichText)
        top_layout.addWidget(title_top)
        
        # Bảng Log
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["PROFILE", "Mã SV & Tên", "Thời gian", "Lớp", "Trạng thái"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setShowGrid(False)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.table.setStyleSheet("""
            QTableWidget {
                background-color: transparent;
                border: none;
            }
            QHeaderView::section {
                background-color: transparent;
                color: #0F172A;
                font-weight: 800;
                font-size: 12px;
                border: none;
                border-bottom: 2px solid #E2E8F0;
                padding: 10px 8px;
                text-align: left;
            }
            QTableWidget::item {
                background-color: transparent;
                border-bottom: 1px solid #F1F5F9;
                color: #334155;
                font-size: 13px;
                font-weight: 600;
            }
        """)
        top_layout.addWidget(self.table)
        main_layout.addWidget(top_panel, 5) # Chiếm 50% chiều cao


        # ==========================================
        # BOTTOM PANELS (Left and Right)
        # ==========================================
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(15)

        # ---------- BOTTOM LEFT ----------
        bottom_left_layout = QVBoxLayout()
        bottom_left_layout.setSpacing(15)

        # 1. Identity Card Panel
        identity_panel = QFrame()
        identity_panel.setObjectName("IdentityPanel")
        identity_panel.setStyleSheet("QFrame#IdentityPanel { background-color: #FFFFFF; border-radius: 12px; border: 1px solid #E2E8F0; }")
        identity_panel.setGraphicsEffect(create_shadow())
        identity_layout = QVBoxLayout(identity_panel)
        identity_layout.setContentsMargins(20, 20, 20, 20)

        id_title = QLabel("<b>IDENTITY CARD FOCUS PANEL</b>")
        id_title.setStyleSheet("font-size: 14px; color: #0F172A; border: none;")
        identity_layout.addWidget(id_title)

        id_content_layout = QHBoxLayout()
        
        # Left side info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(15)
        
        lbl_class = QLabel("🏫 Class:         CS201 - Intro to AI")
        lbl_class.setStyleSheet("font-size: 13px; color: #334155; font-weight: bold;")
        lbl_time = QLabel("⏰ Check-in:   14:02:17")
        lbl_time.setStyleSheet("font-size: 13px; color: #334155; font-weight: bold;")
        lbl_type = QLabel("🎓 Type:          Student")
        lbl_type.setStyleSheet("font-size: 13px; color: #334155; font-weight: bold;")
        
        info_layout.addWidget(lbl_class)
        info_layout.addWidget(lbl_time)
        info_layout.addWidget(lbl_type)
        info_layout.addStretch()
        
        id_content_layout.addLayout(info_layout, 1)

        # Right side Card
        card_frame = QFrame()
        card_frame.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 10px;
                border: 1px solid #E2E8F0;
            }
        """)
        card_layout = QVBoxLayout(card_frame)
        card_layout.setContentsMargins(0,0,0,0)
        card_layout.setSpacing(0)
        
        # Card Header
        card_header = QLabel("STUDENT ID")
        card_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_header.setStyleSheet("background-color: #F8FAFC; color: #64748B; font-weight: bold; font-size: 11px; padding: 6px; border-top-left-radius: 10px; border-top-right-radius: 10px; border-bottom: 1px solid #E2E8F0;")
        card_layout.addWidget(card_header)
        
        # Card Body
        card_body = QHBoxLayout()
        card_body.setContentsMargins(15, 15, 15, 15)
        
        self.identity_img_lbl = QLabel()
        self.identity_img_lbl.setFixedSize(60, 80)
        self.identity_img_lbl.setStyleSheet("background-color: #E2E8F0; border-radius: 6px;")
        self.identity_img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.identity_img_lbl.setText("👤") # Placeholder
        
        card_info = QVBoxLayout()
        self.identity_name_lbl = QLabel("PHẠM MINH ĐỨC")
        self.identity_name_lbl.setStyleSheet("font-size: 14px; font-weight: 900; color: #0F172A; border: none;")
        self.identity_id_lbl = QLabel("ID: 21127005")
        self.identity_id_lbl.setStyleSheet("font-size: 12px; color: #64748B; border: none;")
        card_info.addWidget(self.identity_name_lbl)
        card_info.addWidget(self.identity_id_lbl)
        card_info.addStretch()

        card_body.addWidget(self.identity_img_lbl)
        card_body.addLayout(card_info)
        card_layout.addLayout(card_body)
        
        id_content_layout.addWidget(card_frame, 1)
        identity_layout.addLayout(id_content_layout)

        # Verification Status
        veri_layout = QHBoxLayout()
        lbl_info_extra = QLabel("Gia tăng thông tin định danh & Bảo mật")
        lbl_info_extra.setStyleSheet("color: #94A3B8; font-size: 11px; font-style: italic;")
        
        status_checks = QVBoxLayout()
        status_checks.addWidget(QLabel("✅ <span style='color:#64748B;'>Status:</span> <span style='color:#10B981; font-weight:bold;'>Identity Verified</span>"))
        status_checks.addWidget(QLabel("✅ <span style='color:#64748B;'>Liveness:</span> <span style='color:#10B981; font-weight:bold;'>PASS</span>"))
        
        veri_layout.addWidget(lbl_info_extra)
        veri_layout.addStretch()
        veri_layout.addLayout(status_checks)
        identity_layout.addLayout(veri_layout)

        bottom_left_layout.addWidget(identity_panel, 7)

        # 2. Overall System Status
        overall_panel = QFrame()
        overall_panel.setObjectName("OverallPanel")
        overall_panel.setStyleSheet("QFrame#OverallPanel { background-color: #FFFFFF; border-radius: 12px; border: 1px solid #E2E8F0; }")
        overall_panel.setGraphicsEffect(create_shadow())
        overall_layout = QVBoxLayout(overall_panel)
        overall_layout.setContentsMargins(20, 15, 20, 15)

        overall_title = QLabel("<b>OVERALL SYSTEM STATUS</b>")
        overall_title.setStyleSheet("font-size: 12px; color: #0F172A; border: none;")
        overall_layout.addWidget(overall_title)

        status_row = QHBoxLayout()
        status_row.setSpacing(20)

        # Lưu các label thành biến (self.xxx) để cập nhật Real-time
        self.lbl_overall_cpu = QLabel("🔲 CPU: <b>--%</b>")
        self.lbl_overall_cpu.setStyleSheet("color: #334155; font-size: 12px;")
        
        self.lbl_overall_ram = QLabel("🟩 RAM: <b>--GB / --GB</b>")
        self.lbl_overall_ram.setStyleSheet("color: #334155; font-size: 12px;")
        
        self.lbl_overall_db = QLabel("🗄️ DB Connections: <b>Connected</b>")
        self.lbl_overall_db.setStyleSheet("color: #334155; font-size: 12px;")
        
        self.lbl_overall_uptime = QLabel("⏱️ AI Model Uptime: <b>--h</b>")
        self.lbl_overall_uptime.setStyleSheet("color: #334155; font-size: 12px;")
        
        self.lbl_overall_alert = QLabel("⚠️ Last Alert: <b>None</b>")
        self.lbl_overall_alert.setStyleSheet("color: #334155; font-size: 12px;")

        status_row.addWidget(self.lbl_overall_cpu)
        status_row.addWidget(self.lbl_overall_ram)
        status_row.addWidget(self.lbl_overall_db)
        status_row.addWidget(self.lbl_overall_uptime)
        status_row.addWidget(self.lbl_overall_alert)
        status_row.addStretch()

        overall_layout.addLayout(status_row)
        bottom_left_layout.addWidget(overall_panel, 3)

        # ---------- BOTTOM RIGHT ----------
        bottom_right_layout = QVBoxLayout()
        bottom_right_layout.setSpacing(15)

        # 3. Chart Panel
        chart_panel = QFrame()
        chart_panel.setObjectName("ChartPanel")
        chart_panel.setStyleSheet("QFrame#ChartPanel { background-color: #FFFFFF; border-radius: 12px; border: 1px solid #E2E8F0; }")
        chart_panel.setGraphicsEffect(create_shadow())
        chart_layout_main = QVBoxLayout(chart_panel)
        chart_layout_main.setContentsMargins(20, 20, 20, 20)

        # --- Header Layout cho Chart (Tiêu đề + Bộ chọn ngày) ---
        chart_header_layout = QHBoxLayout()
        
        stat_title = QLabel("<b>CLASS ATTENDANCE STATISTICS</b>")
        stat_title.setTextFormat(Qt.TextFormat.RichText)
        stat_title.setStyleSheet("border: none; font-size: 13px; color: #0F172A;")
        
        # Bộ chọn ngày (Date Picker)
        self.date_picker = QDateEdit()
        self.date_picker.setCalendarPopup(True)
        self.date_picker.setDate(QDate.currentDate()) # Mặc định là ngày hôm nay
        self.date_picker.setDisplayFormat("dd/MM/yyyy")
        self.date_picker.setFixedWidth(130)
        self.date_picker.setStyleSheet("""
            QDateEdit {
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                padding: 4px 8px;
                background: #F8FAFC;
                color: #0F172A;
                font-weight: bold;
                font-size: 12px;
            }
            QDateEdit::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #CBD5E1;
            }
        """)
        # Bắt sự kiện khi người dùng thay đổi ngày
        self.date_picker.dateChanged.connect(self._fetch_stats_by_date)

        chart_header_layout.addWidget(stat_title)
        chart_header_layout.addStretch()
        chart_header_layout.addWidget(self.date_picker)
        
        chart_layout_main.addLayout(chart_header_layout)

        # --- Biểu đồ Donut ---
        chart_layout = QHBoxLayout()
        self.donut = CircularProgress("#3B82F6", "#8B5CF6") # Xanh dương và Tím
        chart_layout.addStretch()
        chart_layout.addWidget(self.donut)
        chart_layout.addStretch()
        chart_layout_main.addLayout(chart_layout)

        # --- Chú thích (Legend) ---
        self.legend = QLabel("🔵 Đã có mặt (0)   🟣 Vắng mặt (0)")
        self.legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.legend.setStyleSheet("font-size: 12px; font-weight: bold; color: #334155; border: none;")
        chart_layout_main.addWidget(self.legend)

        bottom_right_layout.addWidget(chart_panel, 6)

        # 4. Health Monitor Panel
        health_panel = QFrame()
        health_panel.setObjectName("HealthPanel")
        health_panel.setStyleSheet("QFrame#HealthPanel { background-color: #FFFFFF; border-radius: 12px; border: 1px solid #E2E8F0; }")
        health_panel.setGraphicsEffect(create_shadow())
        health_layout = QVBoxLayout(health_panel)
        health_layout.setContentsMargins(20, 15, 20, 15)

        health_title = QLabel("<b>SYSTEM HEALTH MONITOR</b> <span style='font-size:11px; color:#64748B;'>Tình trạng hệ thống</span>")
        health_title.setTextFormat(Qt.TextFormat.RichText)
        health_title.setStyleSheet("border: none; font-size: 13px; color: #0F172A;")
        health_layout.addWidget(health_title)

        grid = QGridLayout()
        grid.setSpacing(10)

        # CPU
        cpu_lbl = QLabel("💻 CPU:")
        cpu_lbl.setStyleSheet("border: none; font-weight: 800; font-size: 11px;")
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setValue(45)
        self.cpu_bar.setFormat("45% Load")
        self.cpu_bar.setFixedHeight(14)
        self.cpu_bar.setStyleSheet("""
            QProgressBar { border: none; background: #E2E8F0; border-radius: 7px; text-align: center; color: white; font-weight: bold; font-size: 9px; }
            QProgressBar::chunk { background-color: #8B5CF6; border-radius: 7px; }
        """)
        grid.addWidget(cpu_lbl, 0, 0)
        grid.addWidget(self.cpu_bar, 0, 1)

        # RAM
        ram_lbl = QLabel("🧠 RAM:")
        ram_lbl.setStyleSheet("border: none; font-weight: 800; font-size: 11px;")
        self.ram_bar = QProgressBar()
        self.ram_bar.setValue(26)
        self.ram_bar.setFormat("2.4 GB / 8 GB")
        self.ram_bar.setFixedHeight(14)
        self.ram_bar.setStyleSheet("""
            QProgressBar { border: none; background: #E2E8F0; border-radius: 7px; text-align: center; color: white; font-weight: bold; font-size: 9px; }
            QProgressBar::chunk { background-color: #3B82F6; border-radius: 7px; }
        """)
        grid.addWidget(ram_lbl, 0, 2)
        grid.addWidget(self.ram_bar, 0, 3)

        # Connections & DB
        conn_lbl = QLabel("🔗 Connections")
        conn_lbl.setStyleSheet("border: none; font-weight: 800; font-size: 11px;")
        self.conn_val = QLabel("🟢 Online")
        self.conn_val.setStyleSheet("border: none; font-size: 11px; color: #10B981; font-weight: bold;")
        
        db_lbl = QLabel("🗄️ Database")
        db_lbl.setStyleSheet("border: none; font-weight: 800; font-size: 11px;")
        self.db_val = QLabel("🟢 Connected")
        self.db_val.setStyleSheet("border: none; font-size: 11px; color: #10B981; font-weight: bold;")
        
        grid.addWidget(conn_lbl, 1, 0)
        grid.addWidget(self.conn_val, 1, 1)
        grid.addWidget(db_lbl, 1, 2)
        grid.addWidget(self.db_val, 1, 3)

        health_layout.addLayout(grid)
        bottom_right_layout.addWidget(health_panel, 4)

        # Add left/right to bottom layout
        bottom_layout.addLayout(bottom_left_layout, 6) # 60%
        bottom_layout.addLayout(bottom_right_layout, 4) # 40%

        main_layout.addLayout(bottom_layout, 5) # Chiếm 50% chiều cao

        # Gọi hàm lấy dữ liệu điểm danh lần đầu tiên khi mở App
        self._fetch_stats_by_date(self.date_picker.date())

    def _create_avatar_label(self, name):
        """Tạo icon Avatar giả lập hình tròn từ chữ cái đầu"""
        initials = "".join([word[0] for word in name.split()[:2]]).upper()
        lbl = QLabel(initials)
        lbl.setFixedSize(30, 30)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("""
            background-color: #E2E8F0; 
            color: #475569; 
            border-radius: 15px; 
            font-weight: bold; 
            font-size: 11px;
        """)
        return lbl

    def _populate_dummy_data(self):
        data = [
            ("21127005", "Đoạn Bình Thọ", "14:02:15", "CS201", True),
            ("21127006", "Nguyễn Trí Ram", "14:02:17", "CS201", True),
            ("21127004", "Nguyễn Thương", "14:02:17", "CS201", True),
            ("21127005", "Phạm Minh Đức", "14:02:28", "CS201", True),
            ("21127005", "Phạm Minh Đức", "14:02:17", "CS201", True),
            ("21127006", "Nguyễn Tiến", "14:02:17", "CS201", True),
            ("21127004", "Nguyễn Thương", "14:02:21", "CS201", True),
        ]
        
        self.table.setRowCount(len(data))
        for row, (uid, name, time, cls, is_present) in enumerate(data):
            # Avatar
            avatar_widget = QWidget()
            avatar_layout = QHBoxLayout(avatar_widget)
            avatar_layout.setContentsMargins(10, 0, 0, 0)
            avatar_layout.addWidget(self._create_avatar_label(name))
            avatar_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.table.setCellWidget(row, 0, avatar_widget)
            
            # Name & ID
            name_item = QTableWidgetItem(f"{uid}   {name}")
            self.table.setItem(row, 1, name_item)
            
            # Time
            time_item = QTableWidgetItem(f"🕒 {time}")
            self.table.setItem(row, 2, time_item)
            
            # Class
            cls_item = QTableWidgetItem(f"📖 {cls}")
            self.table.setItem(row, 3, cls_item)
            
            # Status badge (cột cuối trống theo ảnh, nhưng nếu cần có thể add badge)
            # Dựa theo hình, cột cuối không có dữ liệu, nhưng tôi sẽ thêm badge cho đồng bộ
            # Bạn có thể bỏ đoạn badge này nếu muốn giống ảnh 100%
            '''
            badge_widget = QWidget()
            badge_layout = QHBoxLayout(badge_widget)
            badge_layout.setContentsMargins(0, 0, 0, 0)
            badge_layout.addWidget(StatusBadge("Verified", True))
            badge_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.table.setCellWidget(row, 4, badge_widget)
            '''
            
            self.table.setRowHeight(row, 45)

    def _check_system_realtime(self):
        # 1. Cập nhật CPU & RAM
        try:
            cpu_percent = psutil.cpu_percent()
            # Cập nhật Health Monitor (Bên phải)
            self.cpu_bar.setValue(int(cpu_percent))
            self.cpu_bar.setFormat(f"{cpu_percent:.1f}% Load")
            # Cập nhật Overall System (Bên trái)
            if hasattr(self, 'lbl_overall_cpu'):
                self.lbl_overall_cpu.setText(f"🔲 CPU: <b>{cpu_percent:.1f}%</b>")

            mem = psutil.virtual_memory()
            used_gb = mem.used / (1024**3)
            total_gb = mem.total / (1024**3)
            # Cập nhật Health Monitor (Bên phải)
            self.ram_bar.setValue(int(mem.percent))
            self.ram_bar.setFormat(f"{used_gb:.1f} GB / {total_gb:.1f} GB")
            # Cập nhật Overall System (Bên trái)
            if hasattr(self, 'lbl_overall_ram'):
                self.lbl_overall_ram.setText(f"🟩 RAM: <b>{used_gb:.1f}GB / {total_gb:.1f}GB</b>")
        except:
            pass

        # 2. GỌI CẬP NHẬT BIỂU ĐỒ THEO ĐÚNG NGÀY ĐANG CHỌN TRÊN DATE PICKER

        if hasattr(self, 'date_picker'):
            self._fetch_stats_by_date(self.date_picker.date())

    def add_attendance_log(self, name: str, student_id: str, time_str: str, class_name: str, is_present: bool = True):
        """Thêm một dòng điểm danh mới vào trên cùng của bảng Log"""
        self.table.insertRow(0) # Chèn vào dòng đầu tiên (đẩy các dòng cũ xuống)
        
        # 1. Cột Avatar
        avatar_widget = QWidget()
        avatar_layout = QHBoxLayout(avatar_widget)
        avatar_layout.setContentsMargins(10, 0, 0, 0)
        avatar_layout.addWidget(self._create_avatar_label(name))
        avatar_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.table.setCellWidget(0, 0, avatar_widget)
        
        # 2. Cột Mã SV & Tên
        name_item = QTableWidgetItem(f"{student_id}   {name}")
        self.table.setItem(0, 1, name_item)
        
        # 3. Cột Thời gian
        time_item = QTableWidgetItem(f"🕒 {time_str}")
        self.table.setItem(0, 2, time_item)
        
        # 4. Cột Lớp
        cls_item = QTableWidgetItem(f"📖 {class_name}")
        self.table.setItem(0, 3, cls_item)
        
        # 5. Cột Trạng thái (Thêm Badge xanh Verified)
        badge_widget = QWidget()
        badge_layout = QHBoxLayout(badge_widget)
        badge_layout.setContentsMargins(0, 0, 0, 0)
        badge = StatusBadge("Verified" if is_present else "Failed", is_present)
        badge_layout.addWidget(badge)
        badge_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.table.setCellWidget(0, 4, badge_widget)
        
        self.table.setRowHeight(0, 45)
        
        # Xóa bớt dòng cũ nếu bảng quá dài (Ví dụ: giữ tối đa 50 dòng log gần nhất)
        if self.table.rowCount() > 50:
            self.table.removeRow(50)
            
    def update_stats(self, **kwargs):
        if "students" in kwargs:
            self._students_total = kwargs["students"]
        if "present" in kwargs:
            self._present_count = kwargs["present"]
        if "absent" in kwargs:
            self._absent_count = kwargs["absent"]
            
        # Luôn cập nhật lại biểu đồ, kể cả khi Total = 0
        self.donut.set_value(self._present_count, self._students_total)
        self.legend.setText(f"🔵 Đã có mặt ({self._present_count})   🟣 Vắng mặt ({self._absent_count})")

    def update_latest_snapshot(self, frame_or_pixmap, student_name: str, student_id: str):
        """
        Gắn ảnh và thông tin vào thẻ IDENTITY CARD FOCUS PANEL.
        Hàm này được trigger từ AI Engine (main_window).
        """
        self.identity_name_lbl.setText(student_name.upper() if student_name else "UNKNOWN")
        self.identity_id_lbl.setText(f"ID: {student_id}" if student_id else "ID: ---")
        
        pixmap = None
        if isinstance(frame_or_pixmap, QPixmap):
            pixmap = frame_or_pixmap
        elif isinstance(frame_or_pixmap, QImage):
            pixmap = QPixmap.fromImage(frame_or_pixmap)
        elif isinstance(frame_or_pixmap, np.ndarray):
            rgb = cv2.cvtColor(frame_or_pixmap, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
        if pixmap and not pixmap.isNull():
            # Scale và crop ảnh cho vừa thẻ
            scaled_pixmap = pixmap.scaled(
                self.identity_img_lbl.size(), 
                Qt.AspectRatioMode.KeepAspectRatioByExpanding, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.identity_img_lbl.setPixmap(scaled_pixmap)
            self.identity_img_lbl.setStyleSheet("border-radius: 6px;") # Bỏ màu nền khi đã có ảnh
    def update_system_status(self, key: str, ok: bool, text: str = "", custom_color: str = None):
        """
        Cập nhật trạng thái hệ thống (Database, Camera, AI...) từ main_window
        """
        # Xác định màu và icon dựa trên trạng thái ok
        color = custom_color if custom_color else ("#10B981" if ok else "#EF4444")
        icon = "🟢" if ok else "🔴"
        display_text = f"{icon} {text}"
        
        # Cập nhật UI tương ứng với key
        if key == "database" and hasattr(self, 'db_val'):
            self.db_val.setText(display_text)
            self.db_val.setStyleSheet(f"border: none; font-size: 11px; color: {color}; font-weight: bold;")
            
        elif key == "camera" and hasattr(self, 'conn_val'):
            self.conn_val.setText(display_text)
            self.conn_val.setStyleSheet(f"border: none; font-size: 11px; color: {color}; font-weight: bold;")

    def _fetch_stats_by_date(self, qdate):
        """Hàm gọi API lấy thống kê điểm danh của một ngày cụ thể"""
        date_str = qdate.toString("yyyy-MM-dd") # Format gửi lên API
        
        try:
            # Cập nhật URL này trỏ đến API thực tế của Server bạn
            # Truyền tham số date=YYYY-MM-DD để Server query database
            url = f"http://127.0.0.1:9696/api/dashboard/stats?date={date_str}"
            resp = requests.get(url, headers={"X-DEVICE-TOKEN": "faceattend_secret_2026"}, timeout=2)
            
            if resp.status_code == 200:
                data = resp.json()
                # Giả sử cấu trúc JSON trả về có dạng:
                # {"student_count": 85, "present_count": 72, "absent_count": 13}
                total = data.get("student_count", 0)
                present = data.get("present_count", 0)
                absent = data.get("absent_count", 0)
                
                self.update_stats(students=total, present=present, absent=absent)
            else:
                # Xử lý hiển thị 0 nếu ngày này không có session điểm danh
                self.update_stats(students=0, present=0, absent=0)
                
        except Exception as e:
            logger.error(f"Error fetching stats for {date_str}: {e}")
            # Lỗi kết nối thì đưa thống kê về 0
            self.update_stats(students=0, present=0, absent=0)