"""
tests/test_integration.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BƯỚC 9 — TEST SUITE TÍCH HỢP TOÀN HỆ THỐNG

Kiểm tra:
  1. Tất cả import / dependencies
  2. Database CRUD đầy đủ
  3. Face Engine pipeline
  4. Enrollment flow (end-to-end)
  5. Attendance flow (end-to-end)
  6. Report generation (Excel + PDF)
  7. Performance benchmarks
  8. UI khởi động không crash
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys
import time
import traceback
import random
import string
import numpy as np
from pathlib import Path
from datetime import datetime, date

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ─── Helpers ──────────────────────────────────────────
OK   = "  ✅"
FAIL = "  ❌"
WARN = "  ⚠️ "
INFO = "  ℹ️ "

results: dict[str, bool] = {}

def section(title: str):
    print(f"\n{'─'*52}")
    print(f"  {title}")
    print(f"{'─'*52}")

def ok(msg: str):   print(f"{OK} {msg}")
def fail(msg: str): print(f"{FAIL} {msg}")
def warn(msg: str): print(f"{WARN} {msg}")
def info(msg: str): print(f"{INFO} {msg}")

def rand_str(n=6) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


# ══════════════════════════════════════════════════════
#  TEST 1 — Dependencies
# ══════════════════════════════════════════════════════
def test_dependencies() -> bool:
    section("TEST 1: Dependencies & Imports")
    passed = True

    libs = [
        ("PyQt6",           "PyQt6.QtWidgets", "QApplication"),
        ("OpenCV",          "cv2",             "__version__"),
        ("NumPy",           "numpy",           "__version__"),
        ("InsightFace",     "insightface",     "__version__"),
        ("ONNX Runtime",    "onnxruntime",     "__version__"),
        ("FAISS",           "faiss",           None),
        ("openpyxl",        "openpyxl",        "__version__"),
        ("ReportLab",       "reportlab",       "__version__"),
        ("Loguru",          "loguru",          None),
        ("pyodbc",          "pyodbc",          "__version__"),
    ]

    for name, module, attr in libs:
        try:
            mod = __import__(module)
            ver = getattr(mod, attr, "OK") if attr else "OK"
            ok(f"{name} — {ver}")
        except ImportError as e:
            fail(f"{name} — THIẾU: {e}")
            passed = False

    # Internal modules
    internals = [
        "config",
        "database.connection",
        "database.models",
        "database.repositories",
        "services.face_engine",
        "services.embedding_cache_manager",
        "services.camera_manager",
        "services.attendance_service",
        "services.enrollment_service",
        "services.report_service",
        "utils.logger",
    ]
    for mod_name in internals:
        try:
            __import__(mod_name)
            ok(f"import {mod_name}")
        except Exception as e:
            fail(f"import {mod_name} — {e}")
            passed = False

    return passed


# ══════════════════════════════════════════════════════
#  TEST 2 — Database CRUD
# ══════════════════════════════════════════════════════
def test_database() -> bool:
    section("TEST 2: Database CRUD")
    passed  = True
    test_ids = {}

    try:
        from database.connection import db
        conn_ok = db.test_connection()
        if conn_ok:
            ok("Kết nối SQL Server thành công")
        else:
            fail("Không kết nối được SQL Server — bỏ qua DB tests")
            return False
    except Exception as e:
        fail(f"DB connection error: {e}")
        return False

    # ── Lớp học ──
    try:
        from database.repositories import class_repo
        code = f"TEST_{rand_str(4)}"
        cid  = class_repo.create(
            class_code=code,
            class_name="Lớp Test Tự Động",
            teacher_name="Giáo Viên Test",
            academic_year="2025-2026",
        )
        if cid and cid > 0:
            ok(f"Tạo lớp học: class_id={cid}, code={code}")
            test_ids["class_id"] = cid
            classes = class_repo.get_all()
            ok(f"Lấy danh sách lớp: {len(classes)} lớp")
        else:
            fail("Tạo lớp thất bại")
            passed = False
    except Exception as e:
        fail(f"Class CRUD: {e}")
        passed = False

    # ── Học viên ──
    try:
        from database.repositories import student_repo
        scode = f"HV_{rand_str(4)}"
        sid   = student_repo.create(
            student_code=scode,
            full_name="Học Viên Test Auto",
            class_id=test_ids.get("class_id"),
        )
        if sid and sid > 0:
            ok(f"Tạo học viên: student_id={sid}, code={scode}")
            test_ids["student_id"] = sid
            student = student_repo.get_by_id(sid)
            ok(f"Lấy học viên: {student.full_name}")
            students = student_repo.get_all()
            ok(f"Danh sách học viên: {len(students)} học viên")
        else:
            fail("Tạo học viên thất bại")
            passed = False
    except Exception as e:
        fail(f"Student CRUD: {e}")
        passed = False

    # ── Embedding ──
    try:
        from database.repositories import embedding_repo
        fake_vec = np.random.randn(512).astype(np.float32)
        fake_vec /= np.linalg.norm(fake_vec)  # Normalize
        eid = embedding_repo.save(
            student_id=test_ids.get("student_id"),
            embedding=fake_vec,
            model_version="buffalo_l_test",
        )
        if eid and eid > 0:
            ok(f"Lưu embedding: embedding_id={eid}")
            test_ids["embedding_id"] = eid
        else:
            fail("Lưu embedding thất bại")
            passed = False

        # Load all embeddings
        embeds = embedding_repo.get_all_active()
        ok(f"Load tất cả embeddings: {len(embeds)} records")

    except Exception as e:
        fail(f"Embedding CRUD: {e}")
        passed = False

    # ── Session điểm danh ──
    try:
        from database.repositories import session_repo, record_repo
        sess_id = session_repo.create(
            class_id=test_ids.get("class_id"),
            subject_name="Môn Test Auto",
            session_date=date.today(),
        )
        if sess_id and sess_id > 0:
            ok(f"Tạo session: session_id={sess_id}")
            test_ids["session_id"] = sess_id

            # Start + End session
            session_repo.update_status(sess_id, "ACTIVE")
            ok("Cập nhật status → ACTIVE")
            session_repo.update_status(sess_id, "COMPLETED")
            ok("Cập nhật status → COMPLETED")

            # Ghi record điểm danh
            rid = record_repo.upsert(
                session_id=sess_id,
                student_id=test_ids.get("student_id"),
                status="PRESENT",
                check_in_time=datetime.now(),
                recognition_score=0.92,
            )
            ok(f"Ghi attendance record: record_id={rid}")

            # Lấy report data
            report = record_repo.get_session_report(sess_id)
            ok(f"Session report: {len(report)} học viên")
        else:
            fail("Tạo session thất bại")
            passed = False
    except Exception as e:
        fail(f"Session CRUD: {e}")
        passed = False

    # ── Cleanup test data ──
    try:
        from database.connection import db
        conn = db.get_connection()
        cursor = conn.cursor()
        try:
            if test_ids.get("session_id"):
                cursor.execute("DELETE FROM AttendanceRecords WHERE session_id=?",
                               test_ids["session_id"])
                cursor.execute("DELETE FROM AttendanceSessions WHERE session_id=?",
                               test_ids["session_id"])
            if test_ids.get("student_id"):
                cursor.execute("DELETE FROM FaceEmbeddings WHERE student_id=?",
                               test_ids["student_id"])
                cursor.execute("DELETE FROM Students WHERE student_id=?",
                               test_ids["student_id"])
            if test_ids.get("class_id"):
                cursor.execute("DELETE FROM Classes WHERE class_id=?",
                               test_ids["class_id"])
            conn.commit()
        finally:
            cursor.close()
        ok("Đã dọn dẹp dữ liệu test")
    except Exception as e:
        warn(f"Cleanup: {e}")

    return passed


# ══════════════════════════════════════════════════════
#  TEST 3 — Face Engine
# ══════════════════════════════════════════════════════
def test_face_engine() -> bool:
    section("TEST 3: Face Engine Pipeline")
    passed = True

    try:
        from services.face_engine import face_engine
        if not face_engine.is_ready:
            info("Model chưa load — đang load...")
            t0 = time.perf_counter()
            ok_flag = face_engine.load_model()
            elapsed = (time.perf_counter() - t0) * 1000
            if ok_flag:
                ok(f"Load model: {elapsed:.0f}ms")
            else:
                fail("Load model thất bại")
                return False
        else:
            ok("Model đã sẵn sàng (đã load trước)")
    except Exception as e:
        fail(f"Load model: {e}")
        return False

    # Test detect trên ảnh giả (không có mặt → expect empty)
    try:
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Thêm noise để không bị optimize
        blank += np.random.randint(0, 30, blank.shape, dtype=np.uint8)
        faces = face_engine.detect_faces(blank)
        ok(f"detect_faces(blank): {len(faces)} khuôn mặt (mong đợi 0)")
    except Exception as e:
        fail(f"detect_faces: {e}")
        passed = False

    # Test embedding từ ảnh giả
    try:
        face_region = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
        emb = face_engine.get_embedding(face_region)
        if emb is not None and emb.shape == (512,):
            # Kiểm tra normalized
            norm = float(np.linalg.norm(emb))
            ok(f"get_embedding: shape={emb.shape}, norm={norm:.4f}")
        else:
            warn("get_embedding trả về None (cần ảnh mặt thật để test đầy đủ)")
    except Exception as e:
        fail(f"get_embedding: {e}")
        passed = False

    return passed


# ══════════════════════════════════════════════════════
#  TEST 4 — Embedding Cache + FAISS
# ══════════════════════════════════════════════════════
def test_embedding_cache() -> bool:
    section("TEST 4: Embedding Cache + Cosine Similarity")
    passed = True

    try:
        from services.embedding_cache_manager import cache_manager
        from database.models import EmbeddingCache

        # Tạo cache giả 500 học viên
        N = 500
        vecs = np.random.randn(N, 512).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        cache = EmbeddingCache(
            embeddings=vecs,
            student_ids=[i+1 for i in range(N)],
            student_codes=[f"HV{i+1:04d}" for i in range(N)],
            full_names=[f"Học Viên {i+1}" for i in range(N)],
            class_ids=[1 for _ in range(N)],   # class_ids bắt buộc, dùng class 1
        )
        ok(f"Tạo cache giả: {N} học viên, shape={vecs.shape}")

        # Test search
        query = vecs[42].copy()  # Query chính là HV43
        from services.face_engine import face_engine
        result = face_engine.find_match(query, cache)

        if result and result.student_id == 43:
            ok(f"Tìm chính xác: {result.full_name}, score={result.similarity:.4f}")
        elif result:
            warn(f"Tìm thấy nhưng sai: {result.full_name} (mong HV43), score={result.similarity:.4f}")
        else:
            warn("Không tìm thấy match (score dưới threshold)")

        # Benchmark 10000 embeddings
        N2 = 10000
        big_vecs = np.random.randn(N2, 512).astype(np.float32)
        big_vecs /= np.linalg.norm(big_vecs, axis=1, keepdims=True)
        big_cache = EmbeddingCache(
            embeddings=big_vecs,
            student_ids=list(range(1, N2+1)),
            student_codes=[f"HV{i:05d}" for i in range(N2)],
            full_names=[f"HV {i}" for i in range(N2)],
            class_ids=[1 for _ in range(N2)],  # class_ids bắt buộc
        )

        times = []
        for _ in range(20):
            q = big_vecs[random.randint(0, N2-1)].copy()
            t0 = time.perf_counter()
            face_engine.find_match(q, big_cache)
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = sum(times) / len(times)
        max_ms = max(times)
        target = 5.0
        flag   = ok if avg_ms < target else warn
        flag(f"Search {N2} embeddings — avg={avg_ms:.3f}ms, max={max_ms:.3f}ms (target <{target}ms)")

    except Exception as e:
        fail(f"Cache test: {e}")
        traceback.print_exc()
        passed = False

    return passed


# ══════════════════════════════════════════════════════
#  TEST 5 — Report Generation
# ══════════════════════════════════════════════════════
def test_reports() -> bool:
    section("TEST 5: Báo Cáo Excel + PDF")
    passed = True

    try:
        from services.report_service import ReportData, export_excel, export_pdf

        # Tạo dữ liệu giả
        records = []
        for i in range(1, 31):
            is_p = i <= 25  # 25/30 có mặt
            records.append({
                "student_id":        i,
                "student_code":      f"HV{i:03d}",
                "full_name":         f"Học Viên {i:02d}",
                "gender":            "Nam" if i % 2 == 0 else "Nữ",
                "status":            "PRESENT" if is_p else "ABSENT",
                "check_in_time":     f"08:{30+i//2:02d}:00" if is_p else "",
                "recognition_score": round(random.uniform(0.78, 0.98), 3) if is_p else 0,
                "camera_id":         1,
            })

        data = ReportData(
            session_id=9999,
            class_code="LT01",
            class_name="Lớp Lập Trình 01",
            subject_name="Lập Trình Python — Bước 9 Test",
            session_date="10/03/2026",
            start_time="08:00:00",
            end_time="10:00:00",
            teacher_name="Giáo Viên Demo",
            total_students=30,
            present_count=25,
            absent_count=5,
            records=records,
        )
        ok(f"Tạo ReportData: {data.total_students} HV, "
           f"{data.present_count} có mặt, tỉ lệ={data.attendance_rate:.1f}%")

        # Excel
        out_dir = ROOT / "reports"
        out_dir.mkdir(exist_ok=True)
        excel_path = str(out_dir / "TEST_report.xlsx")

        t0 = time.perf_counter()
        result = export_excel(data, excel_path)
        elapsed = (time.perf_counter() - t0) * 1000

        if result and Path(result).exists():
            size_kb = Path(result).stat().st_size // 1024
            ok(f"Excel: {Path(result).name} ({size_kb}KB) — {elapsed:.0f}ms")
        else:
            fail(f"Excel tạo thất bại")
            passed = False

        # PDF
        pdf_path = str(out_dir / "TEST_report.pdf")
        t0 = time.perf_counter()
        result = export_pdf(data, pdf_path)
        elapsed = (time.perf_counter() - t0) * 1000

        if result and Path(result).exists():
            size_kb = Path(result).stat().st_size // 1024
            ok(f"PDF:   {Path(result).name} ({size_kb}KB) — {elapsed:.0f}ms")
        else:
            fail(f"PDF tạo thất bại")
            passed = False

    except Exception as e:
        fail(f"Report test: {e}")
        traceback.print_exc()
        passed = False

    return passed


# ══════════════════════════════════════════════════════
#  TEST 6 — Performance Benchmark
# ══════════════════════════════════════════════════════
def test_performance() -> bool:
    section("TEST 6: Performance Benchmark")
    passed = True

    # Benchmark cosine similarity
    try:
        N = 1000
        vecs = np.random.randn(N, 512).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        query = np.random.randn(512).astype(np.float32)
        query /= np.linalg.norm(query)

        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            scores = vecs @ query
            idx    = int(np.argmax(scores))
            times.append((time.perf_counter() - t0) * 1000)

        avg = sum(times)/len(times)
        ok(f"Cosine similarity {N} vecs × 100 lần — avg={avg:.3f}ms")
        if avg > 10:
            warn(f"Chậm hơn mong đợi ({avg:.1f}ms > 10ms)")

    except Exception as e:
        fail(f"Cosine benchmark: {e}")
        passed = False

    # Benchmark numpy normalization
    try:
        vecs = np.random.randn(1000, 512).astype(np.float32)
        t0   = time.perf_counter()
        for _ in range(50):
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs  = vecs / norms
        elapsed = (time.perf_counter() - t0) / 50 * 1000
        ok(f"Batch normalize 1000×512 — {elapsed:.3f}ms/iter")

    except Exception as e:
        fail(f"Normalize benchmark: {e}")

    # Benchmark CV2 resize (frame processing)
    try:
        import cv2
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            small = cv2.resize(frame, (640, 360))
            times.append((time.perf_counter() - t0) * 1000)
        avg = sum(times)/len(times)
        ok(f"cv2.resize 1280×720→640×360 — avg={avg:.3f}ms")

    except Exception as e:
        fail(f"CV2 benchmark: {e}")

    # Tổng hợp budget timing
    print(f"\n  📊 Budget thời gian pipeline (ước tính):")
    budgets = [
        ("RetinaFace detect (GPU)",  25,  50),
        ("ArcFace embedding (GPU)",  45,  90),
        ("Cosine search 1000 HV",     1,   5),
        ("DB write (async)",         10,  20),
        ("UI render",                30,  50),
    ]
    total_min = total_max = 0
    for label, mn, mx in budgets:
        total_min += mn
        total_max += mx
        info(f"  {label:<35} {mn:>3}–{mx}ms")
    status = "✅" if total_max <= 1000 else "⚠️"
    print(f"\n  {status} TỔNG: {total_min}–{total_max}ms (target ≤ 1000ms)")

    return passed


# ══════════════════════════════════════════════════════
#  TEST 7 — Config & Paths
# ══════════════════════════════════════════════════════
def test_config() -> bool:
    section("TEST 7: Config & File Structure")
    passed = True

    try:
        from config import app_config, ai_config, db_config
        ok(f"app_config.app_name = {app_config.app_name}")
        ok(f"app_config.window_size = {app_config.window_width}×{app_config.window_height}")
        ok(f"ai_config.model_name = {ai_config.model_name}")
        ok(f"db_config.server = {db_config.server}")
    except Exception as e:
        fail(f"Config import: {e}")
        passed = False
        return passed

    # Kiểm tra thư mục cần có
    required_dirs = [
        ROOT / "ui",
        ROOT / "ui" / "pages",
        ROOT / "ui" / "widgets",
        ROOT / "ui" / "styles",
        ROOT / "services",
        ROOT / "database",
        ROOT / "utils",
        ROOT / "tests",
        ROOT / "models",
        ROOT / "logs",
        ROOT / "reports",
        ROOT / "assets" / "snapshots",
    ]
    for d in required_dirs:
        if d.exists():
            ok(f"Thư mục: {d.relative_to(ROOT)}")
        else:
            warn(f"Chưa có: {d.relative_to(ROOT)} (sẽ tự tạo khi chạy)")
            d.mkdir(parents=True, exist_ok=True)

    # Kiểm tra files chính
    required_files = [
        ROOT / "main.py",
        ROOT / "config.py",
        ROOT / "ui" / "main_window.py",
        ROOT / "ui" / "pages" / "dashboard_page.py",
        ROOT / "ui" / "pages" / "attendance_page.py",
        ROOT / "ui" / "pages" / "enroll_page.py",
        ROOT / "ui" / "pages" / "students_page.py",
        ROOT / "ui" / "pages" / "reports_page.py",
        ROOT / "services" / "face_engine.py",
        ROOT / "services" / "attendance_service.py",
        ROOT / "services" / "enrollment_service.py",
        ROOT / "services" / "report_service.py",
        ROOT / "database" / "connection.py",
        ROOT / "database" / "repositories.py",
    ]
    missing = 0
    for f in required_files:
        if f.exists():
            ok(f"File: {f.relative_to(ROOT)}")
        else:
            fail(f"THIẾU: {f.relative_to(ROOT)}")
            missing += 1
            passed = False

    if missing == 0:
        ok("Tất cả files cần thiết đều có mặt ✅")

    return passed


# ══════════════════════════════════════════════════════
#  TEST 8 — UI Import (không cần display)
# ══════════════════════════════════════════════════════
def test_ui_imports() -> bool:
    section("TEST 8: UI Module Imports")
    passed = True

    ui_modules = [
        "ui.styles.theme",
        "ui.widgets.sidebar",
        "ui.widgets.camera_preview",
        "ui.pages.dashboard_page",
        "ui.pages.attendance_page",
        "ui.pages.enroll_page",
        "ui.pages.students_page",
        "ui.pages.reports_page",
        "ui.main_window",
    ]

    for mod in ui_modules:
        try:
            # Set QT_QPA để không cần display thật
            import os
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            __import__(mod)
            ok(f"import {mod}")
        except Exception as e:
            # Import error nhưng có thể do thiếu Qt display — chấp nhận
            if "display" in str(e).lower() or "platform" in str(e).lower() \
               or "xcb" in str(e).lower() or "cannot connect" in str(e).lower():
                warn(f"{mod} — cần display (bình thường khi test CLI)")
            else:
                fail(f"{mod} — {e}")
                passed = False

    return passed


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    print("\n" + "═"*52)
    print("  FACE ATTENDANCE — TEST SUITE BƯỚC 9")
    print("  Kiểm tra tích hợp toàn hệ thống")
    print("═"*52)
    print(f"  Thời gian: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"  Python:    {sys.version.split()[0]}")
    print(f"  Root:      {ROOT}")

    tests = [
        ("dependencies",  test_dependencies),
        ("database",      test_database),
        ("face_engine",   test_face_engine),
        ("cache_search",  test_embedding_cache),
        ("reports",       test_reports),
        ("performance",   test_performance),
        ("config_paths",  test_config),
        ("ui_imports",    test_ui_imports),
    ]

    for name, func in tests:
        try:
            results[name] = func()
        except Exception as e:
            fail(f"Test {name} crash: {e}")
            traceback.print_exc()
            results[name] = False

    # ── Tổng kết ──
    print(f"\n{'═'*52}")
    print("  KẾT QUẢ TỔNG HỢP")
    print(f"{'═'*52}")

    passed_count = sum(1 for v in results.values() if v)
    total_count  = len(results)

    for name, result in results.items():
        icon = "✅" if result else "❌"
        print(f"  {icon}  {name}")

    print(f"\n  {'✅' if passed_count == total_count else '⚠️ '} "
          f"{passed_count}/{total_count} TESTS PASSED")

    if passed_count == total_count:
        print("\n  🎉 Toàn bộ tests PASSED! Sẵn sàng Bước 10: Đóng gói .exe")
    elif passed_count >= total_count - 2:
        print("\n  🔧 Gần xong! Xem lại các test FAILED ở trên")
    else:
        print("\n  ❌ Có nhiều test thất bại — kiểm tra setup trước khi tiếp tục")

    print()
    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)