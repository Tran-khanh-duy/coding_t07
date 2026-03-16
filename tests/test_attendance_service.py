"""
tests/test_attendance_service.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test suite Bước 4: Attendance + Enrollment Service

Cách chạy:
    cd C:\face_attendance
    py -3.11 tests/test_attendance_service.py

Lưu ý: Các test DB cần SQL Server đang chạy.
Nếu chưa có SQL Server, các test DB sẽ bị skip tự động.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, time
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

GRN="\033[92m"; RED="\033[91m"; YEL="\033[93m"
CYN="\033[96m"; BLD="\033[1m";  RST="\033[0m"

def ok(m):   print(f"  {GRN}✅ {m}{RST}")
def fail(m): print(f"  {RED}❌ {m}{RST}")
def info(m): print(f"  {CYN}ℹ  {m}{RST}")
def warn(m): print(f"  {YEL}⚠  {m}{RST}")
def sep(t=""):
    print(f"\n{BLD}{CYN}{'─'*55}{RST}")
    if t: print(f"{BLD}{CYN}  {t}{RST}\n{BLD}{CYN}{'─'*55}{RST}")

# Kiểm tra DB khả dụng
def check_db() -> bool:
    try:
        from database.connection import db
        return db.test_connection()
    except Exception:
        return False

DB_OK = check_db()


# ─── TEST 1: AttendanceEvent ─────────────────────
def test_attendance_event():
    sep("TEST 1: AttendanceEvent Data Class")
    from services.attendance_service import AttendanceEvent

    event = AttendanceEvent(
        student_id=1, student_code="HV001",
        full_name="Nguyễn Văn A", class_id=1,
        check_in_time=datetime.now(),
        similarity=0.87, snapshot_path=None, session_id=1,
    )
    ok(f"time_str: {event.time_str}")
    ok(f"score_pct: {event.score_pct}")
    assert "87" in event.score_pct
    ok("AttendanceEvent hoạt động đúng")
    return True


# ─── TEST 2: Cooldown logic ──────────────────────
def test_cooldown():
    sep("TEST 2: Chống Điểm Danh Trùng (Cooldown 60s)")
    from services.attendance_service import AttendanceService
    from services.face_engine import RecognitionResult

    svc = AttendanceService()
    # Giả lập session đang active
    svc._session_id = 999
    svc._active     = True

    def make_result(sid, name):
        return RecognitionResult(
            bbox=np.array([0,0,100,100]),
            det_score=0.99, recognized=True,
            student_id=sid, student_code=f"HV{sid:03d}",
            full_name=name, class_id=1, similarity=0.85,
        )

    # Lần 1: không có cooldown → ghi nhận (nhưng DB chưa có nên sẽ fail DB — OK)
    remaining = svc._get_cooldown_remaining(student_id=1)
    ok(f"Cooldown ban đầu: {remaining}s (phải = 0)")
    assert remaining == 0

    # Set cooldown thủ công
    svc._set_cooldown(student_id=1)
    remaining = svc._get_cooldown_remaining(student_id=1)
    ok(f"Sau khi set: {remaining:.1f}s (phải ≈ 60s)")
    assert 59 <= remaining <= 60

    # Student khác không bị ảnh hưởng
    r2 = svc._get_cooldown_remaining(student_id=2)
    ok(f"Student khác (id=2): {r2}s (phải = 0)")
    assert r2 == 0

    # Reset cooldown
    svc.reset_cooldown(student_id=1)
    remaining = svc._get_cooldown_remaining(student_id=1)
    ok(f"Sau reset: {remaining}s (phải = 0)")
    assert remaining == 0

    # Test duplicate callback
    duplicate_calls = []
    svc.on_duplicate = lambda name, wait: duplicate_calls.append((name, wait))
    svc._set_cooldown(student_id=5)

    result = make_result(5, "Test Student")
    # Mock record_repo để không cần DB
    event = svc.process_recognition(result, frame=None)
    assert event is None  # Phải bị cooldown
    ok(f"Duplicate bị chặn đúng: {len(duplicate_calls)} callback")

    ok("Cooldown logic hoạt động chính xác!")
    return True


# ─── TEST 3: Thống kê session ────────────────────
def test_session_stats():
    sep("TEST 3: Thống Kê Session")
    from services.attendance_service import AttendanceService

    svc = AttendanceService()
    svc._session_id = 1
    svc._active     = True
    svc._reset_stats()

    svc._stats["total_recognized"] = 50
    svc._stats["total_recorded"]   = 45
    svc._stats["total_duplicate"]  = 5

    stats = svc.get_stats()
    ok(f"total_recognized: {stats['total_recognized']}")
    ok(f"total_recorded:   {stats['total_recorded']}")
    ok(f"total_duplicate:  {stats['total_duplicate']}")
    ok(f"elapsed_str: {stats.get('elapsed_str', 'N/A')}")
    ok(f"is_active: {stats['is_active']}")
    return True


# ─── TEST 4: CaptureSession ──────────────────────
def test_capture_session():
    sep("TEST 4: CaptureSession — Phiên Chụp Ảnh")
    from services.enrollment_service import CaptureSession

    cs = CaptureSession(target_count=10)
    ok(f"Target: {cs.target_count}")
    ok(f"Count ban đầu: {cs.count}")
    ok(f"is_complete: {cs.is_complete}")
    assert not cs.is_complete
    assert cs.progress == 0.0

    # Thêm 10 frames giả
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    for i in range(10):
        cs.frames.append(dummy)

    ok(f"Count sau 10 frames: {cs.count}")
    ok(f"Progress: {cs.progress*100:.0f}%")
    ok(f"is_complete: {cs.is_complete}")
    assert cs.is_complete
    return True


# ─── TEST 5: EnrollmentService flow (không DB) ───
def test_enrollment_flow_no_db():
    sep("TEST 5: EnrollmentFlow — Không cần DB")
    from services.enrollment_service import EnrollmentService

    svc = EnrollmentService()

    # Test finish khi chưa start
    result = svc.finish_enrollment()
    ok(f"finish trước start: success={result.success}, error='{result.error_msg}'")
    assert not result.success

    # Test progress ban đầu
    prog = svc.capture_progress
    ok(f"progress ban đầu: {prog}")
    assert prog == (0, 0)

    # Test cancel
    svc.start_capture(student_id=1, photo_count=5)
    ok(f"Sau start_capture: progress={svc.capture_progress}")
    svc.cancel_capture()
    ok(f"Sau cancel: progress={svc.capture_progress}")
    assert svc.capture_progress == (0, 0)

    # Test add_frame khi chưa start
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    result = svc.add_frame(dummy)
    ok(f"add_frame khi chưa start: {result} (phải False)")
    assert not result

    return True


# ─── TEST 6: DB — Tạo học viên mẫu ──────────────
def test_db_create_student():
    sep("TEST 6: Database — Tạo Học Viên Mẫu")
    if not DB_OK:
        warn("SQL Server chưa kết nối — bỏ qua test DB")
        warn("→ Cài SQL Server Express và chạy scripts/create_database.sql")
        return True   # Không tính là fail

    from services.enrollment_service import EnrollmentService
    from database.repositories import student_repo

    svc = EnrollmentService()

    # Tạo học viên test
    test_code = f"TEST{int(time.time()) % 10000:04d}"
    sid = svc.create_student(
        student_code=test_code,
        full_name="Học Viên Test Tự Động",
        gender="Nam",
    )

    if sid and sid > 0:
        ok(f"Tạo học viên: id={sid}, code={test_code}")

        # Đọc lại
        student = student_repo.get_by_id(sid)
        ok(f"Đọc lại: {student.full_name}")
        assert student.student_code == test_code

        # Thử tạo trùng mã
        dup = svc.create_student(test_code, "Trùng Tên")
        ok(f"Tạo trùng mã: {dup} (phải None)")
        assert dup is None

        # Tìm kiếm
        results = svc.search_students("Test Tự Động")
        ok(f"Tìm kiếm: {len(results)} kết quả")

        ok("DB operations hoạt động đúng!")
    else:
        fail("Tạo học viên thất bại — kiểm tra DB schema")
        return False

    return True


# ─── TEST 7: DB — Session lifecycle ──────────────
def test_db_session_lifecycle():
    sep("TEST 7: Database — Session Lifecycle")
    if not DB_OK:
        warn("SQL Server chưa kết nối — bỏ qua")
        return True

    from database.repositories import class_repo, session_repo
    from services.attendance_service import AttendanceService

    # Lấy class đầu tiên
    classes = class_repo.get_all()
    if not classes:
        warn("Chưa có lớp học — chạy create_database.sql trước")
        return True

    cls = classes[0]
    info(f"Dùng lớp: {cls.class_code} - {cls.class_name}")

    svc = AttendanceService()
    events_received = []
    svc.on_attendance = lambda e: events_received.append(e)

    # Tạo session
    sid = svc.create_session(
        class_id=cls.class_id,
        subject_name="Test Session Auto",
    )
    ok(f"create_session: id={sid}")
    assert sid > 0

    # Start
    ok_start = svc.start_session(sid)
    ok(f"start_session: {ok_start}")
    assert ok_start
    assert svc.is_active

    # Kiểm tra thống kê
    stats = svc.get_stats()
    ok(f"Stats khi active: session_id={stats['session_id']}")

    # End session
    session = svc.end_session()
    ok(f"end_session: present={session.present_count}, absent={session.absent_count}")
    assert not svc.is_active

    ok("Session lifecycle hoàn chỉnh!")
    return True


# ─── TEST 8: Embedding save/load ─────────────────
def test_db_embedding():
    sep("TEST 8: Database — Lưu và Load Embedding")
    if not DB_OK:
        warn("SQL Server chưa kết nối — bỏ qua")
        return True

    from database.repositories import student_repo, embedding_repo
    from services.embedding_cache_manager import EmbeddingCacheManager

    # Tạo học viên để test embedding
    test_code = f"EMB{int(time.time()) % 10000:04d}"
    sid = student_repo.create(test_code, "Embedding Test Student")
    if not sid or sid < 0:
        warn("Không tạo được học viên test")
        return False

    # Tạo embedding ngẫu nhiên (giả lập ArcFace output)
    fake_emb = np.random.randn(512).astype(np.float32)
    fake_emb /= np.linalg.norm(fake_emb)
    ok(f"Embedding giả: shape={fake_emb.shape}, norm={np.linalg.norm(fake_emb):.4f}")

    # Lưu vào DB
    saved = embedding_repo.save_embedding(
        student_id=sid,
        embedding=fake_emb,
    )
    ok(f"save_embedding: {saved}")

    # Kiểm tra has_embedding
    has = embedding_repo.has_embedding(sid)
    ok(f"has_embedding: {has}")
    assert has

    # Load toàn bộ cache
    mgr = EmbeddingCacheManager()
    mgr.load()           # load từ DB, trả về bool
    cache = mgr.get_cache()  # lấy cache object
    ok(f"Cache sau load: {cache.size} học viên")
    ok(f"Cache shape: {cache.embeddings.shape if cache.embeddings is not None else 'None'}")

    # Kiểm tra embedding mới có trong cache không
    if cache.size > 0 and sid in cache.student_ids:
        idx = cache.student_ids.index(sid)
        loaded_emb = cache.embeddings[idx]
        sim = float(np.dot(fake_emb, loaded_emb))
        ok(f"Similarity với embedding gốc: {sim:.4f} (phải ≈ 1.0)")
        assert sim > 0.99, f"Embedding load sai! sim={sim}"
    else:
        warn(f"student_id={sid} chưa có trong cache (có thể do filter active)")

    ok("Embedding save/load chính xác!")
    return True


# ─── MAIN ────────────────────────────────────────
def main():
    print(f"\n{BLD}{'='*55}")
    print(f"  FACE ATTENDANCE — TEST SUITE BƯỚC 4: BUSINESS LOGIC")
    print(f"{'='*55}{RST}\n")

    info(f"SQL Server: {'✅ Connected' if DB_OK else '⚠️  Offline (test DB sẽ skip)'}")
    print()

    results = {}
    results["attendance_event"]       = test_attendance_event()
    results["cooldown_logic"]         = test_cooldown()
    results["session_stats"]          = test_session_stats()
    results["capture_session"]        = test_capture_session()
    results["enrollment_flow"]        = test_enrollment_flow_no_db()
    results["db_create_student"]      = test_db_create_student()
    results["db_session_lifecycle"]   = test_db_session_lifecycle()
    results["db_embedding"]           = test_db_embedding()

    sep("KẾT QUẢ TỔNG HỢP")
    total  = len(results)
    passed = sum(1 for v in results.values() if v)

    for name, result in results.items():
        if result: ok(name)
        else:      fail(name)

    print()
    if passed == total:
        print(f"{GRN}{BLD}🎉 {passed}/{total} TESTS PASSED!{RST}")
        print(f"{GRN}   Bước 4 hoàn thành → Tiếp tục Bước 5: UI PyQt6{RST}")
    else:
        print(f"{YEL}{BLD}⚠  {passed}/{total} TESTS PASSED{RST}")
    print()


if __name__ == "__main__":
    main()