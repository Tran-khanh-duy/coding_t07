"""
services/report_service.py
Xuất báo cáo điểm danh — Excel + PDF.
"""
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from loguru import logger

from core.config import app_config, report_config


# ─────────────────────────────────────────────
@dataclass
class ReportData:
    session_id:     int
    class_code:     str
    class_name:     str
    subject_name:   str
    session_date:   str
    start_time:     str
    end_time:       str
    teacher_name:   str
    total_students: int
    present_count:  int
    absent_count:   int
    late_count:     int = 0
    records:        list = None

    @property
    def attendance_rate(self) -> float:
        if self.total_students == 0:
            return 0.0
        return self.present_count / self.total_students * 100

    @property
    def title(self) -> str:
        return f"BẢNG ĐIỂM DANH — {self.subject_name}"


# ─────────────────────────────────────────────
def load_report_data(session_id: int) -> Optional[ReportData]:
    """Load dữ liệu điểm danh 1 buổi từ DB."""
    try:
        from database.repositories import session_repo, record_repo, class_repo

        session = session_repo.get_by_id(session_id)
        if not session:
            logger.error(f"Không tìm thấy session {session_id}")
            return None

        records = record_repo.get_session_report(session_id)
        present = [r for r in records if r.get("status") == "PRESENT"]
        absent  = [r for r in records if r.get("status") == "ABSENT"]

        # Lấy teacher_name từ Classes — session không join trực tiếp
        teacher_name = ""
        cls = class_repo.get_by_id(session.class_id)
        if cls:
            teacher_name = cls.teacher_name or ""

        data = ReportData(
            session_id=session_id,
            class_code=getattr(session, "class_code", "") or "",
            class_name=getattr(session, "class_name", "") or "",
            subject_name=session.subject_name,
            session_date=(
                session.session_date.strftime("%d/%m/%Y")
                if session.session_date else ""
            ),
            start_time=(
                session.start_time.strftime("%H:%M:%S")
                if session.start_time else "—"
            ),
            end_time=(
                session.end_time.strftime("%H:%M:%S")
                if session.end_time else "—"
            ),
            teacher_name=teacher_name,
            # total_students = len(records) — không dùng session.total_students
            total_students=len(records),
            present_count=len(present),
            absent_count=len(absent),
            records=records,
        )
        return data

    except Exception as e:
        logger.error(f"load_report_data error: {e}")
        return None


# ─────────────────────────────────────────────
#  Excel Report
# ─────────────────────────────────────────────
def export_excel(data: ReportData, output_path: str = None) -> Optional[str]:
    try:
        import pandas as pd
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

        # 1. Chuyển đổi dữ liệu sang Pandas DataFrame
        records = data.records or []
        if not records:
            df = pd.DataFrame(columns=["student_code", "full_name", "class_name", "gender", "status", "check_in_time", "building"])
        else:
            df = pd.DataFrame(records)

        # Đảm bảo các cột cần thiết tồn tại
        for col in ["building", "class_name", "status"]:
            if col not in df.columns:
                df[col] = "Khác"

        # Khởi tạo Workbook
        wb = Workbook()
        
        # Định dạng chung
        border_thin = Border(
            left=Side(style='thin', color="CCCCCC"), right=Side(style='thin', color="CCCCCC"),
            top=Side(style='thin', color="CCCCCC"), bottom=Side(style='thin', color="CCCCCC")
        )
        header_fill = PatternFill(start_color="2A313C", end_color="2A313C", fill_type="solid") # Dark Gray cho header bảng
        header_font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
        title_fill = PatternFill(start_color="69E29C", end_color="69E29C", fill_type="solid") # Màu Xanh lá (Mint)
        title_font = Font(name="Arial", size=14, bold=True, color="FFFFFF")
        center_align = Alignment(horizontal="center", vertical="center")
        left_align = Alignment(horizontal="left", vertical="center")

        # ── 1. SHEET TỔNG HỢP (DASHBOARD) ──
        ws_dash = wb.active
        ws_dash.title = "Tổng hợp"
        
        ws_dash.merge_cells("A1:E1")
        c = ws_dash["A1"]
        c.value = "TỔNG HỢP ĐIỂM DANH THEO TÒA NHÀ"
        c.font = title_font
        c.fill = title_fill
        c.alignment = center_align
        ws_dash.row_dimensions[1].height = 35

        # Group theo Tòa nhà
        if not df.empty:
            df_bld = df.groupby('building').agg(
                Tổng_SV=('student_code', 'count'),
                Có_mặt=('status', lambda x: (x == 'PRESENT').sum()),
                Vắng=('status', lambda x: (x == 'ABSENT').sum())
            ).reset_index()
            df_bld['Tỉ_lệ_%'] = (df_bld['Có_mặt'] / df_bld['Tổng_SV'] * 100).round(1)
        else:
            df_bld = pd.DataFrame(columns=["building", "Tổng_SV", "Có_mặt", "Vắng", "Tỉ_lệ_%"])

        dash_headers = ["Tòa nhà", "Tổng số SV", "Có mặt", "Vắng mặt", "Tỉ lệ (%)"]
        for col_num, header in enumerate(dash_headers, 1):
            cell = ws_dash.cell(row=3, column=col_num)
            cell.value = header
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center_align
            cell.border = border_thin

        for r_idx, row in enumerate(df_bld.itertuples(index=False), 4):
            for c_idx, val in enumerate(row, 1):
                cell = ws_dash.cell(row=r_idx, column=c_idx)
                cell.value = val
                cell.alignment = center_align
                cell.border = border_thin

        ws_dash.column_dimensions['A'].width = 25
        ws_dash.column_dimensions['B'].width = 15
        ws_dash.column_dimensions['C'].width = 15
        ws_dash.column_dimensions['D'].width = 15
        ws_dash.column_dimensions['E'].width = 15

        # ── 2. CÁC SHEET LỚP (Tạo sheet theo từng nhóm class_name) ──
        if df.empty:
            ws_empty = wb.create_sheet("Trống")
            ws_empty["A1"] = "Không có dữ liệu điểm danh"
        else:
            grouped = df.groupby("class_name")
            for c_name, group in grouped:
                # Tên sheet tối đa 31 ký tự, không chứa ký tự đặc biệt
                sheet_name = str(c_name)[:31].replace(":", "").replace("/", "").replace("\\", "").replace("?", "").replace("*", "").replace("[", "").replace("]", "")
                ws = wb.create_sheet(sheet_name)
                ws.sheet_view.showGridLines = False
                
                # Header Tiêu đề lớn
                ws.merge_cells("A1:F1")
                c = ws["A1"]
                c.value = f" BÁO CÁO ĐIỂM DANH LỚP: {c_name} - NGÀY {data.session_date}"
                c.font = title_font
                c.fill = title_fill
                c.alignment = left_align
                ws.row_dimensions[1].height = 35
                
                # Headers Bảng dữ liệu
                headers = ["STT", "Họ và tên", "MSSV", "Giới tính", "Thời gian", "Ghi chú"]
                for col_num, header in enumerate(headers, 1):
                    cell = ws.cell(row=3, column=col_num)
                    cell.value = header
                    cell.font = header_font
                    cell.fill = header_fill
                    cell.alignment = center_align
                    cell.border = border_thin

                ws.column_dimensions['A'].width = 8
                ws.column_dimensions['B'].width = 30
                ws.column_dimensions['C'].width = 15
                ws.column_dimensions['D'].width = 12
                ws.column_dimensions['E'].width = 15
                ws.column_dimensions['F'].width = 15

                # Render Data Rows
                for idx, row in enumerate(group.itertuples(), 1):
                    is_p = row.status == "PRESENT"
                    ghi_chu = "Có mặt" if is_p else "Vắng"
                    time_str = str(row.check_in_time) if is_p else ""
                    
                    vals = [
                        idx,
                        row.full_name,
                        row.student_code,
                        row.gender,
                        time_str,
                        ghi_chu
                    ]
                    
                    row_num = 3 + idx
                    ws.row_dimensions[row_num].height = 25
                    for col_num, val in enumerate(vals, 1):
                        cell = ws.cell(row=row_num, column=col_num)
                        cell.value = val
                        cell.border = border_thin
                        
                        if col_num in [1, 3, 4, 5]: # STT, MSSV, Giới tính, Thời gian
                            cell.alignment = center_align
                        else: # Họ tên, Ghi chú
                            cell.alignment = left_align
                            if col_num == 2:
                                cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)
                            
                        # Ghi chú (Màu sắc)
                        if col_num == 6:
                            # Chỉnh màu chữ cho "Ghi chú" (Có mặt = Xanh, Vắng = Xám nhạt/Đỏ)
                            cell.font = Font(color="2E7D32" if is_p else "9E9E9E", bold=True)
                            cell.alignment = center_align

        # Đường dẫn lưu file
        if not output_path:
            fname = (
                f"BaoCao_DiemDanh_{data.class_code}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            ).replace("/","-").replace("\\","-").replace(" ","_")
            output_path = str(report_config.output_dir / fname)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        wb.save(output_path)
        logger.success(f"Excel saved: {output_path}")
        return output_path

    except ImportError:
        logger.error("Thư viện 'pandas' hoặc 'openpyxl' chưa được cài đặt. Hãy chạy: pip install pandas openpyxl")
        return None
    except Exception as e:
        logger.error(f"export_excel error: {e}")
        return None


# ─────────────────────────────────────────────
#  PDF Report
# ─────────────────────────────────────────────
def export_pdf(data: ReportData, output_path: str = None) -> Optional[str]:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle, Paragraph,
            Spacer, HRFlowable,
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        if not output_path:
            fname = (
                f"BaoCao_{data.class_code}_{data.subject_name[:20]}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            ).replace("/","-").replace("\\","-").replace(" ","_")
            output_path = str(report_config.output_dir / fname)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        CLR_PANEL   = colors.HexColor("#0D1320"); CLR_CARD  = colors.HexColor("#111B2E")
        CLR_CYAN    = colors.HexColor("#06C8E8"); CLR_GREEN = colors.HexColor("#10D98A")
        CLR_RED     = colors.HexColor("#F04060"); CLR_TEXT  = colors.HexColor("#DCE8F8")
        CLR_DIM     = colors.HexColor("#8BA4C0"); CLR_DARK  = colors.HexColor("#4A6080")

        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            leftMargin=1.5*cm, rightMargin=1.5*cm,
            topMargin=1.5*cm,  bottomMargin=1.5*cm,
        )

        def sty(name, **kw):
            return ParagraphStyle(name, **kw)

        S_TITLE   = sty("t", fontSize=18, fontName="Helvetica-Bold",
                        textColor=CLR_CYAN, alignment=TA_CENTER, spaceAfter=4)
        S_SUB     = sty("s", fontSize=13, fontName="Helvetica-Bold",
                        textColor=CLR_TEXT, alignment=TA_CENTER, spaceAfter=8)
        S_SECTION = sty("sec", fontSize=11, fontName="Helvetica-Bold",
                        textColor=CLR_CYAN, alignment=TA_LEFT, spaceBefore=8)
        S_FOOTER  = sty("f", fontSize=8, fontName="Helvetica",
                        textColor=CLR_DARK, alignment=TA_CENTER)
        S_CELL    = sty("c",  fontSize=9, fontName="Helvetica",
                        textColor=CLR_TEXT, alignment=TA_CENTER)
        S_CELL_L  = sty("cl", fontSize=9, fontName="Helvetica",
                        textColor=CLR_TEXT, alignment=TA_LEFT)

        W = A4[0] - 3*cm
        story = []

        story.append(Paragraph("HỆ THỐNG ĐIỂM DANH KHUÔN MẶT", S_TITLE))
        story.append(Paragraph(data.title, S_SUB))
        story.append(HRFlowable(width="100%", thickness=1, color=CLR_CYAN, spaceAfter=8))

        info = [
            ["Lớp học:", f"{data.class_code} — {data.class_name}", "Ngày:",    data.session_date],
            ["Môn học:", data.subject_name,                         "Giờ:",     f"{data.start_time} — {data.end_time}"],
            ["Giáo viên:", data.teacher_name or "—",               "Session:", str(data.session_id)],
        ]
        info_tbl = Table(info, colWidths=[2.2*cm, 7*cm, 2*cm, 4*cm])
        info_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,-1), CLR_PANEL),
            ("TEXTCOLOR",    (0,0), (0,-1),  CLR_DIM),
            ("TEXTCOLOR",    (2,0), (2,-1),  CLR_DIM),
            ("TEXTCOLOR",    (1,0), (1,-1),  CLR_TEXT),
            ("TEXTCOLOR",    (3,0), (3,-1),  CLR_TEXT),
            ("FONTNAME",     (0,0), (-1,-1), "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), 9),
            ("TOPPADDING",   (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",(0,0), (-1,-1), 5),
            ("LEFTPADDING",  (0,0), (-1,-1), 6),
            ("GRID",         (0,0), (-1,-1), 0.3, CLR_DARK),
        ]))
        story.append(info_tbl)
        story.append(Spacer(1, 12))

        rc = "#10D98A" if data.attendance_rate >= 80 else \
             "#F59E0B" if data.attendance_rate >= 60 else "#F04060"
        stats_data = [[
            Paragraph(f"<font color='#8BA4C0' size='8'>TỔNG HỌC VIÊN</font><br/>"
                      f"<font color='#06C8E8' size='22'><b>{data.total_students}</b></font>", S_CELL),
            Paragraph(f"<font color='#8BA4C0' size='8'>CÓ MẶT</font><br/>"
                      f"<font color='#10D98A' size='22'><b>{data.present_count}</b></font>", S_CELL),
            Paragraph(f"<font color='#8BA4C0' size='8'>VẮNG MẶT</font><br/>"
                      f"<font color='#F04060' size='22'><b>{data.absent_count}</b></font>", S_CELL),
            Paragraph(f"<font color='#8BA4C0' size='8'>TỈ LỆ</font><br/>"
                      f"<font color='{rc}' size='22'><b>{data.attendance_rate:.1f}%</b></font>", S_CELL),
        ]]
        st = Table(stats_data, colWidths=[W/4]*4)
        st.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,-1), CLR_CARD),
            ("TOPPADDING",   (0,0), (-1,-1), 10),
            ("BOTTOMPADDING",(0,0), (-1,-1), 10),
            ("LINEABOVE",    (0,0), (-1,0),  1, CLR_CYAN),
            ("LINEBELOW",    (0,-1),(-1,-1), 1, CLR_CYAN),
            ("INNERGRID",    (0,0), (-1,-1), 0.5, CLR_DARK),
            ("ALIGN",        (0,0), (-1,-1), "CENTER"),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(st)
        story.append(Spacer(1, 14))

        story.append(Paragraph("BẢNG ĐIỂM DANH CHI TIẾT", S_SECTION))
        story.append(Spacer(1, 6))

        hdrs = [Paragraph(f"<b>{h}</b>", S_CELL)
                for h in ["STT","Mã HV","Họ và Tên","Trạng thái","Giờ điểm danh","Độ chính xác"]]
        tbl_data = [hdrs]
        for idx, r in enumerate(data.records or [], 1):
            is_p = r.get("status") == "PRESENT"
            st_txt = f'<font color="#10D98A">✓ Có mặt</font>' if is_p \
                     else f'<font color="#F04060">✗ Vắng</font>'
            tbl_data.append([
                Paragraph(str(idx), S_CELL),
                Paragraph(r.get("student_code",""), S_CELL),
                Paragraph(r.get("full_name",""), S_CELL_L),
                Paragraph(st_txt, S_CELL),
                Paragraph(str(r.get("check_in_time","—")), S_CELL),
                Paragraph(f"{r.get('recognition_score',0)*100:.1f}%" if is_p else "—", S_CELL),
            ])

        dt = Table(tbl_data, colWidths=[1*cm,2.2*cm,6*cm,2.8*cm,3*cm,2.5*cm], repeatRows=1)
        dt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  CLR_PANEL),
            ("LINEBELOW",     (0,0), (-1,0),  1.5, CLR_CYAN),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [CLR_CARD, CLR_PANEL]),
            ("GRID",          (0,0), (-1,-1), 0.3, CLR_DARK),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 4),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("ALIGN",         (2,1), (2,-1),  "LEFT"),
        ]))
        story.append(dt)
        story.append(Spacer(1, 20))

        story.append(HRFlowable(width="100%", thickness=0.5, color=CLR_DARK))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            f"Báo cáo tạo lúc {datetime.now().strftime('%H:%M:%S %d/%m/%Y')} "
            f"· Session ID: {data.session_id}",
            S_FOOTER
        ))

        doc.build(story)
        logger.success(f"PDF saved: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"export_pdf error: {e}")
        return None


# ─────────────────────────────────────────────
def generate_report(session_id: int, fmt: str = "both") -> dict:
    data = load_report_data(session_id)
    if not data:
        return {"success": False, "error": f"Không tìm thấy session {session_id}"}
    result = {"success": True, "excel": None, "pdf": None}
    if fmt in ("excel", "both"):
        result["excel"] = export_excel(data)
    if fmt in ("pdf", "both"):
        result["pdf"]   = export_pdf(data)
    if not any([result["excel"], result["pdf"]]):
        result["success"] = False
        result["error"]   = "Xuất file thất bại"
    return result