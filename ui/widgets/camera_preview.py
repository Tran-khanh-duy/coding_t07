"""
ui/widgets/camera_preview.py
Widget hiển thị live camera feed với overlay HUD (Heads-Up Display) hiện đại.
"""
import numpy as np
import cv2
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QColor, 
    QFont, QPen, QBrush, QLinearGradient
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ui.styles.theme import Colors

class CameraPreviewWidget(QWidget):
    """
    Widget hiển thị camera với phong cách Scanner/AI.
    """
    clicked = pyqtSignal()

    def __init__(self, parent=None, placeholder_text: str = "HỆ THỐNG CHƯA KẾT NỐI"):
        super().__init__(parent)
        self._placeholder = placeholder_text
        self._pixmap: QPixmap | None = None
        self._latency_ms: float = 0.0
        self._status_text: str = "OFFLINE"
        self._status_color: str = Colors.RED
        self._show_hud = True
        
        # Hiệu ứng nhấp nháy cho đèn LIVE
        self._blink = True
        self._blink_timer = Qt.Key.Key_0 # Placeholder logic

        self.setMinimumSize(480, 320)
        self.setStyleSheet(f"""
            background-color: #0A0E14;
            border: 2px solid {Colors.BORDER};
            border-radius: 12px;
        """)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # ─── API Cập nhật ─────────────────────────

    def update_frame(self, frame: np.ndarray):
        """Chuyển đổi BGR numpy sang QPixmap và vẽ lại."""
        if frame is None: return
        try:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            # Chuyển BGR sang RGB trực tiếp bằng OpenCV để hiệu năng tốt hơn
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self._pixmap = QPixmap.fromImage(qt_image)
            self.update()
        except Exception as e:
            print(f"Lỗi hiển thị frame: {e}")

    def update_annotated_frame(self, frame: np.ndarray, latency_ms: float = 0):
        self._latency_ms = latency_ms
        self.update_frame(frame)

    def set_status(self, text: str, color: str = Colors.CYAN):
        self._status_text = text.upper()
        self._status_color = color
        self.update()

    # ─── Vẽ Giao diện (HUD) ────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = self.rect()

        # 1. Vẽ nền hoặc Frame Camera
        if self._pixmap:
            scaled = self._pixmap.scaled(
                rect.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding, # Fill đầy widget
                Qt.TransformationMode.SmoothTransformation
            )
            # Cắt phần thừa để bo góc đẹp
            x = (rect.width() - scaled.width()) // 2
            y = (rect.height() - scaled.height()) // 2
            
            # Vẽ ảnh chính
            painter.drawPixmap(x, y, scaled)
            
            # Thêm một lớp phủ mờ nhẹ ở các cạnh để tăng độ tập trung
            grad = QLinearGradient(0, 0, 0, rect.height())
            grad.setColorAt(0, QColor(0, 0, 0, 100))
            grad.setColorAt(0.2, QColor(0, 0, 0, 0))
            grad.setColorAt(0.8, QColor(0, 0, 0, 0))
            grad.setColorAt(1, QColor(0, 0, 0, 150))
            painter.fillRect(rect, QBrush(grad))

            if self._show_hud:
                self._draw_hud(painter, rect)
        else:
            self._draw_placeholder(painter, rect)

        # 2. Vẽ viền bo góc ngoài cùng
        pen = QPen(QColor(Colors.BORDER), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 10, 10)

    def _draw_hud(self, painter, rect):
        """Vẽ các thành phần HUD trang trí."""
        # --- A. Góc Scanner (4 góc) ---
        pen_corner = QPen(QColor(Colors.CYAN), 3)
        painter.setPen(pen_corner)
        m = 20 # Margin
        len_l = 30 # Độ dài cạnh góc
        
        # Top-Left
        painter.drawLine(m, m, m + len_l, m)
        painter.drawLine(m, m, m, m + len_l)
        # Top-Right
        painter.drawLine(rect.width() - m, m, rect.width() - m - len_l, m)
        painter.drawLine(rect.width() - m, m, rect.width() - m, m + len_l)
        # Bottom-Left
        painter.drawLine(m, rect.height() - m, m + len_l, rect.height() - m)
        painter.drawLine(m, rect.height() - m, m, rect.height() - m - len_l)
        # Bottom-Right
        painter.drawLine(rect.width() - m, rect.height() - m, rect.width() - m - len_l, rect.height() - m)
        painter.drawLine(rect.width() - m, rect.height() - m, rect.width() - m, rect.height() - m - len_l)

        # --- B. LIVE Indicator ---
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(45, 35, "LIVE FEED")
        
        # Đèn nháy đỏ
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(Colors.RED))
        painter.drawEllipse(QPoint(30, 31), 6, 6)

        # --- C. Latency Box (Top Right) ---
        if self._latency_ms > 0:
            lat_text = f"PROC: {self._latency_ms:.0f}ms"
            painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
            tw = painter.fontMetrics().horizontalWidth(lat_text)
            
            # Vẽ nền tối cho chữ
            painter.setBrush(QColor(0, 0, 0, 150))
            painter.drawRoundedRect(rect.width() - tw - 40, 20, tw + 20, 25, 5, 5)
            
            painter.setPen(QColor(Colors.GREEN if self._latency_ms < 150 else Colors.ORANGE))
            painter.drawText(rect.width() - tw - 30, 37, lat_text)

        # --- D. Status Bar (Bottom Left) ---
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Black))
        painter.setPen(QColor(self._status_color))
        painter.drawText(30, rect.height() - 35, f"STATUS: {self._status_text}")

    def _draw_placeholder(self, painter, rect):
        """Vẽ trạng thái khi chưa có camera."""
        painter.fillRect(rect, QColor("#0A0E14"))
        
        # Vẽ họa tiết lưới (Grid) mờ cho cảm giác kỹ thuật
        painter.setPen(QPen(QColor(255, 255, 255, 10), 1))
        for i in range(0, rect.width(), 40):
            painter.drawLine(i, 0, i, rect.height())
        for i in range(0, rect.height(), 40):
            painter.drawLine(0, i, rect.width(), i)

        painter.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        painter.setPen(QColor(Colors.TEXT_DARK))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._placeholder)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()