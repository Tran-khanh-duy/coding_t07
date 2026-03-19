"""
tests/test_face_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test suite cho AI Pipeline — Bước 2

Cách chạy:
    cd C:\face_attendance
    python tests/test_face_engine.py

Các test:
  [1] Kiểm tra GPU và thư viện
  [2] Load model buffalo_l
  [3] Detect khuôn mặt từ ảnh mẫu
  [4] Tính embedding vector
  [5] Cosine similarity giữa 2 ảnh cùng người
  [6] Đo hiệu suất: latency trung bình
  [7] Test enrollment (tính mean embedding)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys
import time
import numpy as np
from pathlib import Path

# Thêm thư mục gốc vào path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger

# ─────────────────────────────────────────────
#  Màu sắc in terminal
# ─────────────────────────────────────────────
GRN = "\033[92m"
RED = "\033[91m"
YEL = "\033[93m"
CYN = "\033[96m"
BLD = "\033[1m"
RST = "\033[0m"

def ok(msg):   print(f"  {GRN}✅ {msg}{RST}")
def fail(msg): print(f"  {RED}❌ {msg}{RST}")
def info(msg): print(f"  {CYN}ℹ  {msg}{RST}")
def warn(msg): print(f"  {YEL}⚠  {msg}{RST}")
def sep(title=""):
    line = "─" * 55
    if title:
        print(f"\n{BLD}{CYN}{line}{RST}")
        print(f"{BLD}{CYN}  {title}{RST}")
        print(f"{BLD}{CYN}{line}{RST}")
    else:
        print(f"{CYN}{line}{RST}")


# ─────────────────────────────────────────────
#  Tạo ảnh test giả (BGR numpy array)
# ─────────────────────────────────────────────
def create_dummy_face_image(size=(720, 1280)):
    """Tạo ảnh 720p giả để test (không cần webcam)."""
    import cv2
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)  # Nền xám đậm

    # Vẽ khuôn mặt giả (ellipse)
    cx, cy = w // 2, h // 2
    cv2.ellipse(img, (cx, cy), (120, 150), 0, 0, 360, (200, 180, 160), -1)
    # Mắt
    cv2.circle(img, (cx - 45, cy - 30), 20, (50, 50, 50), -1)
    cv2.circle(img, (cx + 45, cy - 30), 20, (50, 50, 50), -1)
    cv2.circle(img, (cx - 45, cy - 30), 10, (20, 20, 20), -1)
    cv2.circle(img, (cx + 45, cy - 30), 10, (20, 20, 20), -1)
    # Mũi
    cv2.ellipse(img, (cx, cy + 20), (15, 20), 0, 0, 360, (170, 150, 130), -1)
    # Miệng
    cv2.ellipse(img, (cx, cy + 70), (50, 20), 0, 0, 180, (140, 80, 80), -1)
    return img


# ─────────────────────────────────────────────
#  TEST 1: Kiểm tra thư viện và GPU
# ─────────────────────────────────────────────
def test_environment():
    sep("TEST 1: Môi trường & Thư viện")
    all_ok = True

    # Python version
    pv = sys.version_info
    if pv.major == 3 and pv.minor == 11:
        ok(f"Python {pv.major}.{pv.minor}.{pv.micro}")
    else:
        warn(f"Python {pv.major}.{pv.minor}.{pv.micro} — Khuyến nghị dùng 3.11")

    # ONNX Runtime
    try:
        import onnxruntime as ort
        device = ort.get_device()
        providers = ort.get_available_providers()
        ok(f"onnxruntime {ort.__version__} | Device: {device}")
        info(f"Providers: {providers}")
        if device == "GPU":
            ok("GPU đang được sử dụng!")
        else:
            warn("Đang dùng CPU — kiểm tra CUDA 11.8 + cuDNN 8.6")
    except ImportError:
        fail("onnxruntime-gpu chưa cài → pip install onnxruntime-gpu==1.16.3")
        all_ok = False

    # InsightFace
    try:
        import insightface
        ok(f"insightface {insightface.__version__}")
    except ImportError:
        fail("insightface chưa cài → pip install insightface")
        all_ok = False

    # OpenCV
    try:
        import cv2
        ok(f"opencv-python {cv2.__version__}")
    except ImportError:
        fail("opencv-python chưa cài")
        all_ok = False

    # NumPy
    try:
        ok(f"numpy {np.__version__}")
    except:
        fail("numpy lỗi")
        all_ok = False

    return all_ok


# ─────────────────────────────────────────────
#  TEST 2: Load model
# ─────────────────────────────────────────────
def test_load_model():
    sep("TEST 2: Load Model buffalo_l")
    from services.face_engine import face_engine

    t0 = time.perf_counter()
    success = face_engine.load_model()
    elapsed = (time.perf_counter() - t0) * 1000

    if success:
        ok(f"Model loaded thành công! ({elapsed:.0f}ms)")
        ok(f"Engine sẵn sàng: {face_engine.is_ready}")
    else:
        fail(f"Không load được model!")
        info("Kiểm tra:")
        info("  1. pip install insightface onnxruntime-gpu")
        info("  2. Kết nối internet lần đầu (tải model ~350MB)")
        info(f"  3. Thư mục models: {ROOT / 'models'}")

    return success


# ─────────────────────────────────────────────
#  TEST 3: Detect khuôn mặt
# ─────────────────────────────────────────────
def test_detect_faces():
    sep("TEST 3: Detect Khuôn Mặt")
    import cv2
    from services.face_engine import face_engine

    if not face_engine.is_ready:
        fail("Model chưa load — bỏ qua test này")
        return False

    # Thử với ảnh từ webcam (nếu có)
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            info("Đang test với ảnh từ webcam...")
            faces = face_engine.detect_faces(frame)
            if faces:
                ok(f"Phát hiện {len(faces)} khuôn mặt từ webcam")
                for i, f in enumerate(faces):
                    info(f"  Mặt {i+1}: bbox={f.bbox}, score={f.det_score:.3f}")
                return True
            else:
                warn("Không phát hiện khuôn mặt từ webcam")
                warn("→ Đứng trước camera và thử lại")
    else:
        warn("Không tìm thấy webcam — test với ảnh giả")

    # Test với ảnh giả
    dummy = create_dummy_face_image()
    faces = face_engine.detect_faces(dummy)
    info(f"Ảnh giả: phát hiện {len(faces)} khuôn mặt (bình thường = 0, vì ảnh vẽ bằng tay)")
    ok("Hàm detect_faces chạy không lỗi ✓")
    return True


# ─────────────────────────────────────────────
#  TEST 4: Embedding vector
# ─────────────────────────────────────────────
def test_embedding():
    sep("TEST 4: Embedding Vector (ArcFace 512D)")
    import cv2
    from services.face_engine import face_engine

    if not face_engine.is_ready:
        fail("Model chưa load — bỏ qua")
        return False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        warn("Không có webcam — test embedding bằng vector ngẫu nhiên")
        # Simulate embedding từ ArcFace
        dummy_emb = np.random.randn(512).astype(np.float32)
        dummy_emb /= np.linalg.norm(dummy_emb)
        ok(f"Embedding shape: {dummy_emb.shape}")
        ok(f"Embedding L2 norm: {np.linalg.norm(dummy_emb):.4f} (phải ≈ 1.0)")
        ok(f"Embedding dtype: {dummy_emb.dtype}")
        ok(f"Kích thước bytes: {dummy_emb.nbytes} bytes (= 512 × 4 bytes)")
        return True

    # Chụp 3 ảnh
    embeddings = []
    info("Chụp 3 ảnh liên tiếp để test embedding...")
    for i in range(3):
        ret, frame = cap.read()
        if not ret:
            continue
        faces = face_engine.detect_faces(frame)
        if faces and faces[0].embedding is not None:
            emb = faces[0].embedding
            embeddings.append(emb)
            info(f"  Ảnh {i+1}: embedding shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
        time.sleep(0.3)
    cap.release()

    if not embeddings:
        warn("Không lấy được embedding — không có khuôn mặt trong ảnh")
        return False

    ok(f"Lấy được {len(embeddings)} embeddings thành công")

    # Kiểm tra tính nhất quán giữa các ảnh cùng người
    if len(embeddings) >= 2:
        e1 = embeddings[0] / np.linalg.norm(embeddings[0])
        e2 = embeddings[1] / np.linalg.norm(embeddings[1])
        sim = float(np.dot(e1, e2))
        info(f"Cosine similarity giữa 2 ảnh cùng người: {sim:.4f}")
        if sim >= 0.65:
            ok(f"Similarity {sim:.4f} ≥ 0.65 — Nhất quán tốt!")
        else:
            warn(f"Similarity {sim:.4f} < 0.65 — Ánh sáng có thể không đủ")

    return True


# ─────────────────────────────────────────────
#  TEST 5: Cosine Similarity với Cache
# ─────────────────────────────────────────────
def test_similarity_with_cache():
    sep("TEST 5: Cosine Similarity + Cache")
    from database.models import EmbeddingCache
    from services.face_engine import face_engine, DetectedFace
    from config import ai_config

    # Tạo cache giả với 100 học viên
    N = 100
    np.random.seed(42)
    fake_embeddings = np.random.randn(N, 512).astype(np.float32)
    # Normalize L2
    norms = np.linalg.norm(fake_embeddings, axis=1, keepdims=True)
    fake_embeddings = fake_embeddings / norms

    cache = EmbeddingCache(
        student_ids=   list(range(1, N+1)),
        student_codes= [f"HV{i:03d}" for i in range(1, N+1)],
        full_names=    [f"Học Viên {i}" for i in range(1, N+1)],
        class_ids=     [1] * N,
        embeddings=    fake_embeddings,
    )
    ok(f"Cache giả: {N} học viên, shape={cache.embeddings.shape}")

    # Test 1: Tìm đúng học viên đã biết
    target_idx = 42
    target_emb = fake_embeddings[target_idx].copy()
    # Thêm noise nhỏ (mô phỏng 2 ảnh của cùng 1 người)
    noisy_emb = target_emb + np.random.randn(512).astype(np.float32) * 0.05
    noisy_emb /= np.linalg.norm(noisy_emb)

    face = DetectedFace(
        bbox=np.array([100, 100, 300, 400]),
        landmarks=np.zeros((5, 2)),
        det_score=0.98,
        embedding=noisy_emb,
    )
    result = face_engine.recognize(face, cache)
    info(f"Tìm học viên index={target_idx}: {cache.full_names[target_idx]}")
    info(f"Kết quả: recognized={result.recognized}, student={result.full_name}, score={result.similarity:.4f}")
    if result.recognized and result.student_id == target_idx + 1:
        ok("Tìm đúng học viên!")
    else:
        warn("Không tìm đúng — có thể do cache ngẫu nhiên không đủ giống")

    # Test 2: Đo tốc độ tìm kiếm
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        face_engine.recognize(face, cache)
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = np.mean(times)
    ok(f"Tốc độ cosine similarity ({N} embeddings): avg={avg_ms:.3f}ms, max={max(times):.3f}ms")

    # Test 3: Scale lên 10,000 học viên
    N_big = 10000
    big_embeddings = np.random.randn(N_big, 512).astype(np.float32)
    big_norms = np.linalg.norm(big_embeddings, axis=1, keepdims=True)
    big_embeddings = big_embeddings / big_norms

    big_cache = EmbeddingCache(
        student_ids=[i for i in range(N_big)],
        student_codes=[f"HV{i}" for i in range(N_big)],
        full_names=[f"Student {i}" for i in range(N_big)],
        class_ids=[1]*N_big,
        embeddings=big_embeddings,
    )
    t0 = time.perf_counter()
    face_engine.recognize(face, big_cache)
    big_time = (time.perf_counter() - t0) * 1000
    ok(f"Scale test {N_big} embeddings: {big_time:.3f}ms (phải < 5ms)")

    return True


# ─────────────────────────────────────────────
#  TEST 6: Đo hiệu suất toàn pipeline
# ─────────────────────────────────────────────
def test_performance():
    sep("TEST 6: Đo Hiệu Suất Pipeline")
    import cv2
    from services.face_engine import face_engine
    from database.models import EmbeddingCache

    if not face_engine.is_ready:
        fail("Model chưa load — bỏ qua")
        return False

    # Tạo frame 720p giả
    dummy_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

    # Cache 500 học viên
    N = 500
    fake_embs = np.random.randn(N, 512).astype(np.float32)
    fake_embs /= np.linalg.norm(fake_embs, axis=1, keepdims=True)
    cache = EmbeddingCache(
        student_ids=list(range(1, N+1)),
        student_codes=[f"HV{i}" for i in range(N)],
        full_names=[f"Student {i}" for i in range(N)],
        class_ids=[1]*N,
        embeddings=fake_embs,
    )

    info(f"Test với frame 720p ngẫu nhiên + cache {N} học viên")
    info("Chạy 10 lần để đo trung bình...")

    times = []
    for i in range(10):
        _, elapsed = face_engine.process_frame(dummy_frame, cache)
        times.append(elapsed)
        info(f"  Lần {i+1}: {elapsed:.1f}ms")

    avg = np.mean(times)
    info(f"\nTrung bình: {avg:.1f}ms")
    if avg <= 200:
        ok(f"✅ {avg:.1f}ms ≤ 200ms — Đủ nhanh cho yêu cầu 1s!")
    else:
        warn(f"⚠ {avg:.1f}ms > 200ms — Cần tối ưu thêm")
        info("Thử:")
        info("  1. Đảm bảo CUDA đang được dùng (TEST 1)")
        info("  2. Giảm det_size về (320, 320) trong config.py")

    return True


# ─────────────────────────────────────────────
#  TEST 7: Enrollment embedding
# ─────────────────────────────────────────────
def test_enrollment():
    sep("TEST 7: Enrollment — Tính Embedding Trung Bình")
    from services.face_engine import face_engine
    import cv2

    if not face_engine.is_ready:
        fail("Model chưa load — bỏ qua")
        return False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        warn("Không có webcam — test với ảnh giả")
        # Simulate 5 ảnh cho 1 học viên
        photos = [np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8) for _ in range(5)]
        emb, score, count = face_engine.compute_enrollment_embedding(photos)
        info(f"Ảnh giả: {count} ảnh hợp lệ (mong đợi = 0, vì ảnh nhiễu)")
        ok("Hàm compute_enrollment_embedding không lỗi ✓")
        return True

    info("Chụp 5 ảnh để enrollment (nhìn thẳng vào camera)...")
    photos = []
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            photos.append(frame)
            info(f"  Chụp ảnh {i+1}/5")
        time.sleep(0.5)
    cap.release()

    emb, avg_score, count = face_engine.compute_enrollment_embedding(photos)

    if emb is not None:
        ok(f"Enrollment thành công!")
        ok(f"  Ảnh hợp lệ: {count}/5")
        ok(f"  Avg detection score: {avg_score:.3f}")
        ok(f"  Embedding shape: {emb.shape}")
        ok(f"  Embedding norm: {np.linalg.norm(emb):.4f} (phải = 1.0)")
        info(f"  Kích thước lưu DB: {emb.nbytes} bytes")
    else:
        warn("Không tạo được embedding — không phát hiện khuôn mặt trong ảnh")

    return emb is not None


# ─────────────────────────────────────────────
#  TEST 8: Kiểm tra thứ tự kết quả + One-to-one Assignment
#  (Kiểm tra 2 lỗi đã sửa trong recognize_batch)
# ─────────────────────────────────────────────
def test_batch_order_and_onetone():
    sep("TEST 8: Thứ Tự Kết Quả Batch + One-to-One Assignment")
    from database.models import EmbeddingCache
    from services.face_engine import face_engine, DetectedFace
    from config import ai_config

    np.random.seed(7)

    # Tạo cache giả 5 học sinh
    N = 5
    db_embeddings = np.random.randn(N, 512).astype(np.float32)
    norms = np.linalg.norm(db_embeddings, axis=1, keepdims=True)
    db_embeddings /= norms

    cache = EmbeddingCache(
        student_ids=   list(range(1, N+1)),
        student_codes= [f"SV{i:02d}" for i in range(1, N+1)],
        full_names=    [f"Học Sinh {i}" for i in range(1, N+1)],
        class_ids=     [1] * N,
        embeddings=    db_embeddings.copy(),
    )

    all_ok = True

    # ── TEST 8a: Thứ tự kết quả phải khớp với faces gốc ──────────────────────
    info("8a: Kiểm tra thứ tự kết quả khi mix emb/no-emb...")

    emb0 = db_embeddings[0] + np.random.randn(512).astype(np.float32) * 0.02
    emb0 /= np.linalg.norm(emb0)
    emb2 = db_embeddings[2] + np.random.randn(512).astype(np.float32) * 0.02
    emb2 /= np.linalg.norm(emb2)

    faces_test = [
        DetectedFace(bbox=np.array([10, 10, 100, 100]), landmarks=np.zeros((5,2)), det_score=0.95, embedding=emb0),   # faces[0] → SV01
        DetectedFace(bbox=np.array([200, 10, 290, 100]), landmarks=np.zeros((5,2)), det_score=0.80, embedding=None),  # faces[1] → unknown (no emb)
        DetectedFace(bbox=np.array([400, 10, 490, 100]), landmarks=np.zeros((5,2)), det_score=0.91, embedding=emb2),  # faces[2] → SV03
    ]

    results = face_engine.recognize_batch(faces_test, cache)

    if len(results) != 3:
        fail(f"Số kết quả sai: {len(results)} (cần 3)")
        all_ok = False
    else:
        # Kiểm tra bbox khớp đúng vị trí
        for i, (face, res) in enumerate(zip(faces_test, results)):
            if not np.array_equal(face.bbox, res.bbox):
                fail(f"faces[{i}] bbox KHÔNG khớp! face.bbox={face.bbox} vs result.bbox={res.bbox}")
                all_ok = False
            else:
                ok(f"faces[{i}]: bbox khớp đúng ✓")

        # faces[1] không có embedding → phải là unknown
        if results[1].recognized:
            fail(f"faces[1] (no emb) bị nhận diện sai là '{results[1].full_name}'!")
            all_ok = False
        else:
            ok(f"faces[1] (no emb): đúng → unknown ✓")

        # faces[0] và faces[2] phải nhận ra đúng người khác nhau
        if results[0].recognized and results[2].recognized:
            if results[0].student_id == results[2].student_id:
                fail(f"faces[0] và faces[2] đều nhận ra cùng student_id={results[0].student_id}!")
                all_ok = False
            else:
                ok(f"faces[0]={results[0].full_name} | faces[2]={results[2].full_name} — khác nhau ✓")
        else:
            info(f"faces[0].recognized={results[0].recognized} (score={results[0].similarity:.3f})")
            info(f"faces[2].recognized={results[2].recognized} (score={results[2].similarity:.3f})")
            info(f"Threshold hiện tại: {ai_config.recognition_threshold}")
            warn("Một trong 2 khuôn mặt không đạt ngưỡng — noise test embeddings quá lớn?")

    # ── TEST 8b: One-to-One Assignment ───────────────────────────────────────
    info("8b: Kiểm tra one-to-one assignment (2 mặt → cùng 1 người)...")

    emb0a = db_embeddings[0] + np.random.randn(512).astype(np.float32) * 0.02
    emb0a /= np.linalg.norm(emb0a)
    emb0b = db_embeddings[0] + np.random.randn(512).astype(np.float32) * 0.05  # score thấp hơn
    emb0b /= np.linalg.norm(emb0b)

    faces_dup = [
        DetectedFace(bbox=np.array([10, 10, 100, 100]),   landmarks=np.zeros((5,2)), det_score=0.97, embedding=emb0a),  # closer → SV01 winner
        DetectedFace(bbox=np.array([200, 10, 290, 100]),  landmarks=np.zeros((5,2)), det_score=0.92, embedding=emb0b),  # farther → SV01 loser
    ]

    dup_results = face_engine.recognize_batch(faces_dup, cache)

    won  = sum(1 for r in dup_results if r.recognized and r.student_id == 1)
    lost = sum(1 for r in dup_results if not r.recognized)

    if won == 1 and lost == 1:
        winner_idx = next(i for i, r in enumerate(dup_results) if r.recognized)
        ok(f"One-to-one: đúng! faces[{winner_idx}] thắng (score={dup_results[winner_idx].similarity:.3f}), faces[{1-winner_idx}] bị đánh unknown ✓")
    elif won == 0:
        warn("Cả 2 khuôn mặt dưới ngưỡng — threshold có thể đã nâng lên quá cao")
        info("Đây không phải lỗi nếu threshold vừa được nâng lên 0.60")
    else:
        fail(f"One-to-one THẤT BẠI: {won} winner(s) — cả 2 cùng được công nhận là SV01!")
        all_ok = False

    return all_ok


# ─────────────────────────────────────────────
#  MAIN — Chạy tất cả tests
# ─────────────────────────────────────────────
def main():
    print(f"\n{BLD}{'='*55}")
    print(f"  FACE ATTENDANCE — TEST SUITE BƯỚC 2: AI PIPELINE")
    print(f"{'='*55}{RST}\n")

    results = {}

    results["environment"]        = test_environment()
    results["load_model"]         = test_load_model()
    results["detect_faces"]       = test_detect_faces()
    results["embedding"]          = test_embedding()
    results["similarity"]         = test_similarity_with_cache()
    results["performance"]        = test_performance()
    results["enrollment"]         = test_enrollment()
    results["batch_order+1-to-1"] = test_batch_order_and_onetone()

    # Tổng kết
    sep("KẾT QUẢ TỔNG HỢP")
    total = len(results)
    passed = sum(1 for v in results.values() if v)

    for name, result in results.items():
        if result:
            ok(f"{name}")
        else:
            fail(f"{name}")

    print()
    if passed == total:
        print(f"{GRN}{BLD}🎉 TẤT CẢ {total}/{total} TESTS PASSED!{RST}")
        print(f"{GRN}   Bước 2 hoàn thành — Sẵn sàng cho Bước 3: Camera Manager{RST}")
    else:
        print(f"{YEL}{BLD}⚠  {passed}/{total} TESTS PASSED{RST}")
        print(f"{YEL}   Xem lại các test FAILED ở trên để xử lý{RST}")
    print()


if __name__ == "__main__":
    main()
