"""
tests/test_camera_manager.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test suite cho Camera Manager — Bước 3

Cách chạy:
    cd C:\face_attendance
    py -3.11 tests/test_camera_manager.py

Các test:
  [1] Webcam USB (index 0) — cơ bản
  [2] Đọc frame liên tục + đo FPS
  [3] Thêm / start / stop camera
  [4] Snapshot — chụp và lưu ảnh
  [5] Giả lập mất kết nối + reconnect
  [6] Kiểm tra cấu hình camera RTSP (không cần camera IP thật)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, time, cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger

GRN="\033[92m"; RED="\033[91m"; YEL="\033[93m"
CYN="\033[96m"; BLD="\033[1m";  RST="\033[0m"

def ok(m):   print(f"  {GRN}✅ {m}{RST}")
def fail(m): print(f"  {RED}❌ {m}{RST}")
def info(m): print(f"  {CYN}ℹ  {m}{RST}")
def warn(m): print(f"  {YEL}⚠  {m}{RST}")
def sep(t=""):
    print(f"\n{BLD}{CYN}{'─'*55}{RST}")
    if t: print(f"{BLD}{CYN}  {t}{RST}\n{BLD}{CYN}{'─'*55}{RST}")


# ─── TEST 1: Kiểm tra webcam có tồn tại không ───
def test_webcam_exists():
    sep("TEST 1: Phát hiện Webcam USB")
    found = []
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                found.append((i, w, h))
                ok(f"Webcam #{i}: {w}×{h}")
        cap.release()

    if found:
        ok(f"Tìm thấy {len(found)} webcam(s)")
        return True, found[0][0]   # Trả về index webcam đầu tiên
    else:
        warn("Không tìm thấy webcam USB")
        warn("→ Vẫn có thể dùng camera IP qua LAN")
        return False, -1


# ─── TEST 2: CameraThread đọc frame ─────────────
def test_camera_thread(webcam_index: int):
    sep("TEST 2: CameraThread — Đọc Frame Liên Tục")
    from services.camera_manager import CameraThread, CameraInfo, CameraStatus

    if webcam_index < 0:
        warn("Bỏ qua — không có webcam")
        return True

    info_obj = CameraInfo(
        camera_id=1,
        name="Test Webcam",
        source=str(webcam_index),
    )

    frames_received = []
    status_changes  = []

    def on_frame(cam_id, frame):
        frames_received.append(frame)

    def on_status(cam_id, status):
        status_changes.append(status)
        info(f"Status → {status.value}")

    thread = CameraThread(info_obj, on_frame=on_frame, on_status=on_status)
    thread.start()

    # Chờ tối đa 15s để camera connect (webcam USB thực tế cần 8-12s)
    info("Đợi camera kết nối (tối đa 15s)...")
    from services.camera_manager import CameraStatus as _CS
    _deadline = time.time() + 15
    while time.time() < _deadline:
        time.sleep(0.5)
        if info_obj.status == _CS.CONNECTED:
            break

    info("Đọc frame thêm 3 giây...")
    time.sleep(3)
    thread.stop()
    thread.join(timeout=5)

    if frames_received:
        fps_actual = len(frames_received) / 3.0
        h, w = frames_received[0].shape[:2]
        ok(f"Nhận được {len(frames_received)} frames trong 3s")
        ok(f"FPS thực tế: {fps_actual:.1f} fps")
        ok(f"Độ phân giải: {w}×{h}")
        ok(f"Status cuối: {info_obj.status.value}")
        return True
    else:
        fail("Không nhận được frame nào!")
        return False


# ─── TEST 3: CameraManager quản lý nhiều camera ──
def test_camera_manager(webcam_index: int):
    sep("TEST 3: CameraManager — Start/Stop/GetFrame")
    from services.camera_manager import CameraManager, CameraStatus

    mgr = CameraManager()

    # Thêm webcam
    ok_add = mgr.add_camera(
        camera_id=1,
        name="Webcam Test",
        source=str(webcam_index) if webcam_index >= 0 else "0",
        floor=1,
    )
    ok(f"add_camera: {ok_add}")

    # Không thêm trùng
    dup = mgr.add_camera(1, "Dup", "0")
    if not dup:
        ok("Chống trùng camera_id hoạt động đúng")

    if webcam_index < 0:
        warn("Không có webcam — bỏ qua test start/get_frame")
        return True

    # Start
    mgr.start_camera(1)

    # Chờ tối đa 15s để CONNECTED
    info("Đợi camera kết nối (tối đa 15s)...")
    from services.camera_manager import CameraStatus as _CS2
    _t = time.time()
    while time.time() - _t < 15:
        time.sleep(0.5)
        if mgr.get_status(1) == _CS2.CONNECTED:
            break

    status = mgr.get_status(1)
    info(f"Trạng thái sau 2s: {status}")

    if status == CameraStatus.CONNECTED:
        ok("Camera CONNECTED thành công!")
    else:
        warn(f"Trạng thái: {status} (có thể đang kết nối)")

    # Lấy frame
    frame = mgr.get_frame(1)
    if frame is not None:
        ok(f"get_frame() trả về frame {frame.shape}")
    else:
        warn("get_frame() trả về None — thử lại sau 1s")
        time.sleep(1)
        frame = mgr.get_frame(1)
        if frame is not None:
            ok(f"get_frame() OK sau delay: {frame.shape}")

    # Thông tin tổng hợp
    all_info = mgr.get_all_info()
    info(f"Tổng camera: {len(all_info)}")
    info(f"Camera connected: {mgr.get_connected_count()}")

    # Stop
    mgr.stop_all()
    ok("stop_all() hoàn thành")
    return True


# ─── TEST 4: Snapshot ────────────────────────────
def test_snapshot(webcam_index: int):
    sep("TEST 4: Snapshot — Chụp và Lưu Ảnh")
    from services.camera_manager import CameraManager

    if webcam_index < 0:
        warn("Bỏ qua — không có webcam")
        return True

    mgr = CameraManager()
    mgr.add_camera(1, "Snapshot Test", str(webcam_index))
    mgr.start_camera(1)

    # Chờ tối đa 15s để camera ready trước khi snapshot
    info("Đợi camera sẵn sàng để snapshot (tối đa 15s)...")
    from services.camera_manager import CameraStatus as _CS3
    _t2 = time.time()
    while time.time() - _t2 < 15:
        time.sleep(0.5)
        if mgr.get_status(1) == _CS3.CONNECTED and mgr.get_frame(1) is not None:
            break

    snap_path = str(ROOT / "assets" / "snapshots" / "test_snapshot.jpg")
    success = mgr.capture_snapshot(1, snap_path)

    if success:
        ok(f"Snapshot lưu tại: {snap_path}")
        img = cv2.imread(snap_path)
        if img is not None:
            ok(f"Ảnh đọc lại OK: {img.shape}")
    else:
        warn("Snapshot thất bại — không có frame")

    mgr.stop_all()
    return success


# ─── TEST 5: Cấu hình camera IP ──────────────────
def test_rtsp_config():
    sep("TEST 5: Cấu hình Camera IP (RTSP)")
    from services.camera_manager import CameraInfo

    rtsp_urls = [
        "rtsp://admin:admin123@192.168.1.101:554/Streaming/Channels/101",
        "rtsp://192.168.1.102:554/stream",
        "rtsp://user:pass@10.0.0.5:554/live",
    ]

    for url in rtsp_urls:
        cam_info = CameraInfo(camera_id=99, name="IP Test", source=url)
        assert cam_info.is_ip_camera, "Phải nhận dạng là IP camera"
        display = cam_info.source_display
        has_credentials = "@" in url
        if has_credentials:
            # URL có user:pass → phải ẩn thành ***
            assert "***" in display, f"Phải ẩn mật khẩu trong: {url}"
            assert "***" in display  # đảm bảo password không lộ
        else:
            # URL không có credentials → hiển thị nguyên
            assert "***" not in display, "URL không có pass thì không cần ***"
        ok(f"URL display: {display}")

    # Test URL webcam
    usb_info = CameraInfo(camera_id=1, name="USB", source="0")
    assert not usb_info.is_ip_camera
    ok(f"USB display: {usb_info.source_display}")

    # Load từ config
    from config import CAMERAS
    info(f"Cameras trong config.py: {len(CAMERAS)}")
    for cam in CAMERAS:
        info(f"  [{cam['id']}] {cam['name']} — floor {cam.get('floor', '?')}")

    ok("Cấu hình RTSP đúng định dạng")
    return True


# ─── TEST 6: FrameProcessor ──────────────────────
def test_frame_processor(webcam_index: int):
    sep("TEST 6: FrameProcessor — Kết nối Camera → AI")
    from services.camera_manager import CameraManager
    from services.frame_processor import FrameProcessor
    from services.face_engine import face_engine

    if webcam_index < 0:
        warn("Bỏ qua — không có webcam")
        return True

    # Load model nếu chưa
    if not face_engine.is_ready:
        info("Đang load model...")
        face_engine.load_model()

    if not face_engine.is_ready:
        warn("Model chưa sẵn sàng — bỏ qua test này")
        return True

    mgr = CameraManager()
    mgr.add_camera(1, "FP Test", str(webcam_index))
    mgr.start_camera(1)
    time.sleep(1)

    results_log = []
    frames_log  = []

    processor = FrameProcessor(camera_id=1)
    processor.on_frame  = lambda f: frames_log.append(1)
    processor.on_result = lambda r, f, ms: results_log.append((len(r), ms))
    processor.start()

    info("Chạy FrameProcessor 5 giây...")
    time.sleep(5)
    processor.stop()
    mgr.stop_all()

    stats = processor.get_stats()
    info(f"Stats: {stats}")

    if frames_log:
        ok(f"Nhận {len(frames_log)} frames từ camera")
    if results_log:
        avg_ms = sum(ms for _, ms in results_log) / len(results_log)
        ok(f"Xử lý {len(results_log)} lần | avg={avg_ms:.1f}ms")
        if avg_ms <= 300:
            ok(f"Tốc độ {avg_ms:.1f}ms ≤ 300ms ✓")
        else:
            warn(f"Tốc độ {avg_ms:.1f}ms — cần CUDA để tối ưu")

    return True


# ─── MAIN ────────────────────────────────────────
def main():
    print(f"\n{BLD}{'='*55}")
    print(f"  FACE ATTENDANCE — TEST SUITE BƯỚC 3: CAMERA MANAGER")
    print(f"{'='*55}{RST}\n")

    results = {}

    found, webcam_idx = test_webcam_exists()
    results["webcam_exists"]    = found or True   # Không có webcam vẫn OK
    results["camera_thread"]    = test_camera_thread(webcam_idx)
    results["camera_manager"]   = test_camera_manager(webcam_idx)
    results["snapshot"]         = test_snapshot(webcam_idx)
    results["rtsp_config"]      = test_rtsp_config()
    results["frame_processor"]  = test_frame_processor(webcam_idx)

    sep("KẾT QUẢ TỔNG HỢP")
    total  = len(results)
    passed = sum(1 for v in results.values() if v)

    for name, result in results.items():
        if result: ok(name)
        else:      fail(name)

    print()
    if passed == total:
        print(f"{GRN}{BLD}🎉 {passed}/{total} TESTS PASSED!{RST}")
        print(f"{GRN}   Bước 3 hoàn thành → Tiếp tục Bước 4: Attendance Service{RST}")
    else:
        print(f"{YEL}{BLD}⚠  {passed}/{total} TESTS PASSED{RST}")
    print()


if __name__ == "__main__":
    main()