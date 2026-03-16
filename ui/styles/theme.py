"""
ui/styles/theme.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Light theme hiện đại — Windows 11 Fluent Design style.
Cải tiến: Chống cắt chữ, tối ưu hiển thị nút bấm/ô nhập liệu.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from PyQt6.QtGui import QColor, QPalette


class Colors:
    # ── Nền ──────────────────────────────────
    BG_DARK     = "#EEF2F7"   # App background — xám xanh nhạt
    BG_PANEL    = "#FFFFFF"   # Sidebar / panel
    BG_CARD     = "#FFFFFF"   # Card
    BG_INPUT    = "#F8FAFD"   # Input field
    BG_HOVER    = "#F0F5FF"   # Hover
    BG_SELECTED = "#DBEAFE"   # Selected

    # ── Viền ─────────────────────────────────
    BORDER      = "#E2E8F0"
    BORDER_LT   = "#CBD5E1"

    # ── Accent ───────────────────────────────
    CYAN        = "#2563EB"   # Primary blue
    CYAN_DIM    = "#1D4ED8"
    GREEN       = "#10B981"   # Success
    GREEN_DIM   = "#059669"
    RED         = "#EF4444"   # Error
    RED_DIM     = "#DC2626"
    ORANGE      = "#F59E0B"   # Warning
    PURPLE      = "#8B5CF6"   # Info

    # ── Văn bản ───────────────────────────────
    TEXT        = "#1E293B"
    TEXT_DIM    = "#64748B"
    TEXT_DARK   = "#94A3B8"

    # ── Camera ───────────────────────────────
    CAM_BG      = "#0F172A"
    CAM_ON      = "#10B981"
    CAM_OFF     = "#EF4444"
    CAM_WAIT    = "#F59E0B"


MAIN_STYLESHEET = f"""
/* ── Global ── */
QWidget {{
    background-color: {Colors.BG_DARK};
    color: {Colors.TEXT};
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
}}
QMainWindow {{
    background-color: {Colors.BG_DARK};
}}

/* ── Scrollbar ── */
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    border-radius: 4px;
    margin: 2px 1px;
}}
QScrollBar::handle:vertical {{
    background: {Colors.BORDER_LT};
    border-radius: 4px;
    min-height: 40px;
}}
QScrollBar::handle:vertical:hover {{ background: {Colors.TEXT_DARK}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {Colors.BORDER_LT};
    border-radius: 4px;
}}

/* ── Button ── */
QPushButton {{
    background-color: {Colors.BG_PANEL};
    color: {Colors.TEXT};
    border: 1px solid {Colors.BORDER_LT};
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 14px;
    min-height: 40px;
}}
QPushButton:hover {{
    background-color: {Colors.BG_HOVER};
    border-color: {Colors.CYAN};
    color: {Colors.CYAN};
}}
QPushButton:pressed {{
    background-color: {Colors.BG_SELECTED};
}}
QPushButton:disabled {{
    color: {Colors.TEXT_DARK};
    border-color: {Colors.BORDER};
    background-color: {Colors.BG_INPUT};
}}

/* ── Input ── */
QLineEdit {{
    background-color: {Colors.BG_CARD};
    color: {Colors.TEXT};
    border: 1.5px solid {Colors.BORDER_LT};
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 14px;
    min-height: 40px;
    selection-background-color: {Colors.BG_SELECTED};
    selection-color: {Colors.TEXT};
}}
QLineEdit:focus {{ border-color: {Colors.CYAN}; }}
QLineEdit:disabled {{
    color: {Colors.TEXT_DARK};
    background-color: {Colors.BG_INPUT};
    border-color: {Colors.BORDER};
}}

/* ── ComboBox ── */
QComboBox {{
    background-color: {Colors.BG_CARD};
    color: {Colors.TEXT};
    border: 1.5px solid {Colors.BORDER_LT};
    border-radius: 8px;
    padding: 10px 14px;
    min-height: 40px;
    font-size: 14px;
}}
QComboBox:focus {{ border-color: {Colors.CYAN}; }}
QComboBox:disabled {{
    background-color: {Colors.BG_INPUT};
    color: {Colors.TEXT_DARK};
}}
QComboBox::drop-down {{ border: none; width: 30px; }}
QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {Colors.TEXT_DIM};
    margin-right: 10px;
}}
QComboBox QAbstractItemView {{
    background-color: {Colors.BG_CARD};
    border: 1px solid {Colors.BORDER_LT};
    border-radius: 8px;
    color: {Colors.TEXT};
    selection-background-color: {Colors.BG_HOVER};
    selection-color: {Colors.CYAN};
    outline: none;
    padding: 4px;
    min-height: 30px;
}}

/* ── Label ── */
QLabel {{
    background: transparent;
    color: {Colors.TEXT};
}}

/* ── Table ── */
QTableWidget {{
    background-color: {Colors.BG_CARD};
    border: 1px solid {Colors.BORDER};
    border-radius: 10px;
    gridline-color: {Colors.BORDER};
    color: {Colors.TEXT};
    font-size: 14px;
    alternate-background-color: #FAFBFD;
    selection-background-color: {Colors.BG_SELECTED};
    selection-color: {Colors.CYAN};
}}
QTableWidget::item {{
    padding: 10px 14px;
    border-bottom: 1px solid {Colors.BORDER};
}}
QTableWidget::item:selected {{
    background-color: {Colors.BG_SELECTED};
    color: {Colors.CYAN};
}}
QTableWidget::item:hover {{ background-color: {Colors.BG_HOVER}; }}
QHeaderView::section {{
    background-color: #F8FAFD;
    color: {Colors.TEXT_DIM};
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.8px;
    padding: 12px 14px;
    border: none;
    border-bottom: 2px solid {Colors.BORDER_LT};
}}

/* ── ProgressBar ── */
QProgressBar {{
    background-color: #EEF2F7;
    border: none;
    border-radius: 6px;
    height: 12px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {Colors.CYAN_DIM}, stop:1 #60A5FA);
    border-radius: 6px;
}}

/* ── Splitter ── */
QSplitter::handle {{ background-color: {Colors.BORDER}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}

/* ── Tooltip ── */
QToolTip {{
    background-color: {Colors.TEXT};
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
}}

/* ── CheckBox ── */
QCheckBox {{
    color: {Colors.TEXT};
    spacing: 10px;
}}
QCheckBox::indicator {{
    width: 20px;
    height: 20px;
    border: 2px solid {Colors.BORDER_LT};
    border-radius: 5px;
    background: {Colors.BG_CARD};
}}
QCheckBox::indicator:checked {{
    background: {Colors.CYAN};
    border-color: {Colors.CYAN};
}}
QCheckBox::indicator:hover {{ border-color: {Colors.CYAN}; }}

/* ── SpinBox / DateEdit ── */
QSpinBox, QDateEdit, QTimeEdit {{
    background-color: {Colors.BG_CARD};
    color: {Colors.TEXT};
    border: 1.5px solid {Colors.BORDER_LT};
    border-radius: 8px;
    padding: 8px 12px;
    min-height: 40px;
}}
QSpinBox:focus, QDateEdit:focus {{ border-color: {Colors.CYAN}; }}

/* ── ScrollArea ── */
QScrollArea {{ border: none; background: transparent; }}

/* ── MessageBox ── */
QMessageBox {{ background-color: {Colors.BG_CARD}; }}
QMessageBox QLabel {{
    color: {Colors.TEXT};
    font-size: 14px;
    min-width: 320px;
}}
QMessageBox QPushButton {{
    min-width: 90px;
    min-height: 38px;
    padding: 8px 16px;
}}
"""


def apply_theme(app):
    app.setStyleSheet(MAIN_STYLESHEET)
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(Colors.BG_DARK))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(Colors.TEXT))
    palette.setColor(QPalette.ColorRole.Base,            QColor(Colors.BG_CARD))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor("#FAFBFD"))
    palette.setColor(QPalette.ColorRole.Text,            QColor(Colors.TEXT))
    palette.setColor(QPalette.ColorRole.Button,          QColor(Colors.BG_PANEL))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(Colors.TEXT))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(Colors.CYAN))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(Colors.TEXT_DARK))
    palette.setColor(QPalette.ColorRole.Mid,             QColor(Colors.BORDER))
    palette.setColor(QPalette.ColorRole.Shadow,          QColor("#C0C8D4"))
    app.setPalette(palette)


def card_style(accent: str = None, radius: int = 12) -> str:
    c = accent or Colors.BORDER
    return (
        f"background-color: {Colors.BG_CARD};"
        f"border: 1px solid {c};"
        f"border-radius: {radius}px;"
        f"padding: 18px;"
    )


def badge_style(color: str) -> str:
    return (
        f"background-color: {color}18;"
        f"color: {color};"
        f"border: 1px solid {color}44;"
        f"border-radius: 6px;"
        f"padding: 4px 12px;"
        f"font-size: 12px;"
        f"font-weight: 700;"
    )

def combo_style() -> str:
    return f"""
        QComboBox {{
            background: {Colors.BG_CARD};
            color: {Colors.TEXT};
            border: 1.5px solid {Colors.BORDER_LT};
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 14px;
            min-height: 42px;
        }}
        QComboBox:focus {{ border-color: {Colors.CYAN}; }}
        QComboBox:disabled {{
            background: {Colors.BG_INPUT};
            color: {Colors.TEXT_DARK};
        }}
        QComboBox::drop-down {{ border: none; width: 30px; }}
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid {Colors.TEXT_DIM};
            margin-right: 10px;
        }}
        QComboBox QAbstractItemView {{
            background: {Colors.BG_CARD};
            color: {Colors.TEXT};
            border: 1px solid {Colors.BORDER_LT};
            border-radius: 8px;
            selection-background-color: {Colors.BG_HOVER};
            selection-color: {Colors.CYAN};
            outline: none;
        }}
    """


def input_style() -> str:
    return f"""
        QLineEdit {{
            background: {Colors.BG_CARD};
            color: {Colors.TEXT};
            border: 1.5px solid {Colors.BORDER_LT};
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 14px;
        }}
        QLineEdit:focus {{ border-color: {Colors.CYAN}; }}
        QLineEdit:disabled {{
            background: {Colors.BG_INPUT};
            color: {Colors.TEXT_DARK};
        }}
    """