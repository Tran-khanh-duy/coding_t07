"""
ui/widgets/camera_preview.py
Widget hiển thị live camera feed với overlay thông tin.
"""
import numpy as np
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ui.styles.theme import Colors


class CameraPreviewWidget(QWidget):
    """
    Widget hiển thị frame camera realtime.
    Nhận QImage hoặc numpy BGR array, hiển thị co giãn theo kích thước widget.
    """
    clicked = pyqtSignal()

    def __init__(self, parent=None, placeholder_text: str = "📷  Camera chưa kết nối"):
        super().__init__(parent)
        self._placeholder = placeholder_text
        self._pixmap: QPixmap | None = None
        self._fps: float = 0.0
        self._latency_ms: float = 0.0
        self._status_text: str = ""
        self._status_color: str = Colors.TEXT_DIM

        self.setMinimumSize(320, 240)
        self.setStyleSheet(f"""
            background-color: #050810;
            border: 1px solid {Colors.BORDER};
            border-radius: 10px;
        """)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # ─── Cập nhật frame ───────────────────────

    def update_frame(self, frame: np.ndarray):
        """Nhận numpy BGR array từ camera, chuyển thành pixmap."""
        try:
            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self._pixmap = QPixmap.fromImage(img)
            self.update()
        except Exception:
            pass

    def update_annotated_frame(self, frame: np.ndarray, latency_ms: float = 0):
        """Cập nhật frame đã được vẽ bounding box."""
        self._latency_ms = latency_ms
        self.update_frame(frame)

    def set_status(self, text: str, color: str = Colors.TEXT_DIM):
        self._status_text = text
        self._status_color = color
        self.update()

    def clear(self):
        self._pixmap = None
        self.update()

    # ─── Paint ────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        rect = self.rect()

        if self._pixmap:
            # Scale giữ tỉ lệ, căn giữa
            scaled = self._pixmap.scaled(
                rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (rect.width()  - scaled.width())  // 2
            y = (rect.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

            # Latency overlay
            if self._latency_ms > 0:
                painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
                color = Colors.GREEN if self._latency_ms < 200 else Colors.ORANGE
                painter.setPen(QColor(color))
                painter.drawText(rect.adjusted(10, 10, -10, -10),
                                 Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                                 f"{self._latency_ms:.0f}ms")
        else:
            # Placeholder
            painter.fillRect(rect, QColor("#050810"))
            painter.setFont(QFont("Segoe UI", 14))
            painter.setPen(QColor(Colors.TEXT_DARK))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._placeholder)

        # Status badge dưới cùng
        if self._status_text:
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            painter.setPen(QColor(self._status_color))
            painter.drawText(
                rect.adjusted(10, -30, -10, -8),
                Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
                f"● {self._status_text}",
            )

    def mousePressEvent(self, event):
        self.clicked.emit()
