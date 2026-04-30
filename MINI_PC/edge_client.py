"""
edge_client.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Edge AI Service — Chạy trên Mini PC
Nhiệm vụ:
  1. Kéo embedding vectors từ Server API về RAM
  2. Đọc Camera → Nhận diện khuôn mặt (InsightFace)
  3. Chống giả mạo (Anti-Spoofing)
  4. Gửi kết quả điểm danh về Server qua REST API
  5. Nếu mất mạng → lưu vào SQLite offline queue → tự đồng bộ khi có lại

Khởi động:
    python edge_client.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import time
import base64
import sqlite3
import threading
import requests
import cv2
import os
import numpy as np
from datetime import datetime
from loguru import logger
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Server"))

# Tắt log nhiễu của OpenCV (index out of range)
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

from config import edge_config, ai_config, anti_spoof_config
from database.models import EmbeddingCache
from utils.camera_utils import detect_available_cameras, discover_network_cameras, generate_rtsp_links


class EdgeClient:
    """
    Client kết nối tới Server API.
    Quản lý: Pull embeddings, Push attendance, Offline queue.
    """

    def __init__(self):
        self.server_url = edge_config.server_url.rstrip("/")
        self.api_key = edge_config.api_key
        self.camera_id = edge_config.camera_id

        # Embedding cache cục bộ
        # Global cache cũ (nếu không có camera_id)
        self._cache: EmbeddingCache = EmbeddingCache()
        
        # [NEW] Multi-camera caches
        self._multi_caches = {}  # { camera_id: EmbeddingCache }
        self._multi_cache_times = {} # { camera_id: float (timestamp) }
        
        self._cache_lock = threading.Lock()

        # Trạng thái kết nối
        self._server_online = False
        self._last_embed_pull = 0.0
        
        # Setup Session với Retry logic
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        self._session = requests.Session()
        retries = Retry(
            total=3, 
            backoff_factor=0.5, 
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        self._active_status_cache = {} # Cache trạng thái từ HeadlessProcessor

        # Offline queue (SQLite)
        self._db_path = Path(__file__).parent / "database" / "edge_offline.db"
        self._init_offline_db()

        # Cooldown tracking
        self._cooldown_map: dict[int, float] = {}
        self._cooldown_lock = threading.Lock()

        # [FIX] Theo dõi phiên bản embedding — phát hiện có học viên mới ngay lập tức
        self._known_embedding_version: int = -1
        self._last_version_check: float = 0.0

        # Background sync thread
        self._stop_event = threading.Event()
        self._sync_thread = threading.Thread(
            target=self._sync_loop, name="Edge-Sync", daemon=True
        )
        self._sync_thread.start()

        # [NEW] Background IP Camera Discovery thread
        self._discovered_rtsp = []
        self._discovery_thread = threading.Thread(
            target=self._discovery_loop, name="Edge-Discovery", daemon=True
        )
        self._discovery_thread.start()

        # [NEW] Trạng thái camera sẽ được báo cáo định kỳ trong _sync_loop
        logger.info(f"EdgeClient khởi tạo | Server: {self.server_url} | Camera: {self.camera_id}")

    # ─── Offline DB ───────────────────────────

    def _init_offline_db(self):
        """Khởi tạo SQLite queue cho offline records."""
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS offline_attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        camera_id TEXT,
                        timestamp TEXT,
                        embedding BLOB,
                        liveness_score REAL,
                        liveness_checked INTEGER DEFAULT 0,
                        synced INTEGER DEFAULT 0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            logger.info(f"Offline DB sẵn sàng: {self._db_path}")
        except Exception as e:
            logger.error(f"Lỗi init offline DB: {e}")

    # ─── Server Communication ─────────────────

    def _headers(self) -> dict:
        return {"X-API-Key": self.api_key, "Content-Type": "application/json"}

    def check_server(self) -> bool:
        """Kiểm tra server còn online không."""
        try:
            resp = self._session.get(
                f"{self.server_url}/api/health",
                timeout=3,
            )
            online = (resp.status_code == 200)
            if online and not self._server_online:
                # Vừa kết nối lại -> Báo cáo trạng thái ngay
                logger.info("🌐 Server Online - Đang báo cáo trạng thái...")
                threading.Thread(target=self.report_status, daemon=True).start()
            self._server_online = online
            return online
        except Exception:
            self._server_online = False
            return False

    @property
    def is_server_online(self) -> bool:
        return self._server_online

    # ─── Dynamic Camera Detection ──────────────

    def _get_local_ip(self) -> str:
        """Lấy địa chỉ IP mạng nội bộ của máy."""
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Không cần kết nối thực sự, chỉ để lấy IP interface
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    def report_status(self):
        """Kiểm tra các cổng camera và báo cáo về Server."""
        try:
            status_map = {}

            # [FIX] Gửi cả Tên và Source của Camera
            for cam in edge_config.camera_list:
                is_active = self._active_status_cache.get(cam["id"], False)
                if not is_active:
                    is_active = self._active_status_cache.get(cam["source"], False)
                    
                status_map[cam["id"]] = {
                    "name": cam["name"],
                    "source": cam["source"],
                    "is_active": is_active
                }

            # Bổ sung các camera đang hoạt động khác nếu có
            for cam_id, is_active in self._active_status_cache.items():
                if is_active and cam_id not in status_map:
                    status_map[cam_id] = {
                        "name": f"Auto {cam_id}",
                        "source": cam_id,
                        "is_active": True
                    }

            payload = {
                "device_name": edge_config.device_name,
                "camera_status": status_map,
                "ip_address": self._get_local_ip(), # Vẫn giữ IP Box để tham khảo
                "timestamp": datetime.now().isoformat()
            }
            
            self._session.post(
                f"{self.server_url}/api/system/edge_status",
                json=payload,
                headers=self._headers(),
                timeout=5
            )
            logger.debug(f"📡 Báo cáo trạng thái: {list(status_map.keys())}")
        except Exception as e:
            logger.error(f"Lỗi khi report_status: {e}")

    def update_active_status(self, camera_id: str, is_active: bool):
        """Cập nhật bộ đệm trạng thái từ Processor."""
        self._active_status_cache[camera_id] = is_active

    # ─── Pull Embeddings ──────────────────────

    def pull_embeddings(self, target_camera_id: str = None) -> bool:
        """
        Kéo toàn bộ embedding vectors từ Server về RAM.
        Nếu truyền target_camera_id, sẽ lưu vào cache riêng của camera đó.
        """
        cam_id = target_camera_id or self.camera_id
        try:
            logger.info(f"📥 Đang tải embeddings từ Server cho camera_id={cam_id}...")
            resp = self._session.get(
                f"{self.server_url}/api/embeddings",
                params={"camera_id": cam_id},
                headers=self._headers(),
                timeout=30,
            )

            if resp.status_code != 200:
                logger.error(f"Server trả về lỗi {resp.status_code}: {resp.text}")
                return False

            data = resp.json()
            if data.get("count", 0) == 0:
                logger.warning(f"Server chưa có embedding nào cho {cam_id}!")
                with self._cache_lock:
                    if target_camera_id:
                        self._multi_caches[target_camera_id] = EmbeddingCache()
                        self._multi_cache_times[target_camera_id] = time.time()
                    else:
                        self._cache = EmbeddingCache()
                return True

            # Parse embeddings
            new_cache = EmbeddingCache()
            vecs = []
            for s in data["students"]:
                emb_bytes = base64.b64decode(s["embedding_b64"])
                vec = np.frombuffer(emb_bytes, dtype=np.float32).copy()
                if vec.shape[0] != 512:
                    logger.warning(f"Skip student {s['student_id']}: shape={vec.shape}")
                    continue

                new_cache.student_ids.append(s["student_id"])
                new_cache.student_codes.append(s["student_code"])
                new_cache.full_names.append(s["full_name"])
                new_cache.class_ids.append(s.get("class_id", 0))
                new_cache.class_names.append(s.get("class_name", ""))
                new_cache.class_codes.append(s.get("class_code", ""))
                vecs.append(vec)

            if vecs:
                mat = np.vstack(vecs).astype(np.float32)
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                new_cache.embeddings = mat / np.maximum(norms, 1e-8)

            with self._cache_lock:
                if target_camera_id:
                    self._multi_caches[target_camera_id] = new_cache
                    self._multi_cache_times[target_camera_id] = time.time()
                else:
                    self._cache = new_cache
                    self._last_embed_pull = time.time()

            self._server_online = True
            logger.success(f"✅ Đã tải {new_cache.size} khuôn mặt (Camera {cam_id}) từ Server vào RAM")
            return True

        except requests.ConnectionError:
            logger.warning("❌ Không thể kết nối Server — làm việc offline")
            self._server_online = False
            return False
        except Exception as e:
            logger.error(f"Lỗi pull embeddings: {e}")
            self._server_online = False
            return False

    def get_cache(self, camera_id: str = None) -> EmbeddingCache:
        """Lấy cache hiện tại (thread-safe). Lấy cache riêng của camera nếu có."""
        with self._cache_lock:
            if camera_id and camera_id in self._multi_caches:
                return self._multi_caches[camera_id]
            return self._cache

    def should_refresh_embeddings(self, camera_id: str = None) -> bool:
        """Kiểm tra đã đến lúc refresh embeddings chưa."""
        if camera_id and camera_id in self._multi_cache_times:
            last_pull = self._multi_cache_times[camera_id]
        else:
            last_pull = self._last_embed_pull
            
        elapsed_min = (time.time() - last_pull) / 60
        return elapsed_min >= edge_config.embedding_refresh_min

    # ─── Push Attendance ──────────────────────

    def send_attendance(
        self,
        embedding: np.ndarray,
        camera_id: str = None,
        liveness_score: float = 1.0,
        liveness_checked: bool = False,
    ) -> dict:
        """
        Gửi kết quả nhận diện về Server.
        Nếu mất mạng → lưu offline.
        """
        payload = {
            "camera_id": camera_id or self.camera_id,
            "timestamp": datetime.now().isoformat(),
            "embedding": embedding.tolist(),
            "liveness_score": liveness_score,
            "liveness_checked": liveness_checked,
        }

        try:
            resp = self._session.post(
                f"{self.server_url}/api/attendance",
                json=payload,
                headers=self._headers(),
                timeout=5,
            )

            if resp.status_code == 200:
                result = resp.json()
                self._server_online = True
                return result
            else:
                logger.error(f"Server lỗi {resp.status_code}: {resp.text}")
                self._save_offline(payload, embedding, liveness_score, liveness_checked)
                return {"status": "offline", "message": "Đã lưu offline"}

        except requests.ConnectionError:
            logger.warning("📴 Mất kết nối Server — lưu offline")
            self._server_online = False
            self._save_offline(payload, embedding, liveness_score, liveness_checked)
            return {"status": "offline", "message": "Đã lưu offline"}
        except Exception as e:
            logger.error(f"Lỗi gửi attendance: {e}")
            self._save_offline(payload, embedding, liveness_score, liveness_checked)
            return {"status": "error", "message": str(e)}

    def _save_offline(self, payload, embedding, liveness_score, liveness_checked):
        """Lưu bản ghi vào SQLite khi mất mạng."""
        try:
            emb_bytes = embedding.astype(np.float32).tobytes()
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """INSERT INTO offline_attendance 
                       (camera_id, timestamp, embedding, liveness_score, liveness_checked)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        payload["camera_id"],
                        payload["timestamp"],
                        emb_bytes,
                        liveness_score,
                        1 if liveness_checked else 0,
                    ),
                )
            logger.info("💾 Đã lưu vào offline queue")
        except Exception as e:
            logger.error(f"Lỗi lưu offline: {e}")

    # ─── Cooldown ─────────────────────────────

    def check_cooldown(self, student_id: int, camera_id: str) -> float:
        """Trả về thời gian cooldown còn lại (giây). 0 = hết cooldown."""
        with self._cooldown_lock:
            key = f"{student_id}_{camera_id}"
            last = self._cooldown_map.get(key, 0)
            elapsed = time.time() - last
            remaining = max(0.0, edge_config.attendance_cooldown - elapsed)
            return remaining

    def set_cooldown(self, student_id: int, camera_id: str):
        with self._cooldown_lock:
            key = f"{student_id}_{camera_id}"
            self._cooldown_map[key] = time.time()

    # ─── Background Sync ─────────────────────

    def _sync_loop(self):
        """Vòng lặp nền: đồng bộ offline records + refresh embeddings thông minh."""
        time.sleep(5)  # Chờ khởi động ổn định
        while not self._stop_event.is_set():
            try:
                # 1. Sync offline records
                if self.check_server():
                    self._push_offline_records()
                    
                    # [NEW] Thường xuyên cập nhật danh sách camera động
                    self.report_status()

                    # [FIX] Kiểm tra phiên bản embedding mỗi 10 giây
                    # Thay vì chờ 10 phút, chỉ pull khi có phân bản mới
                    now = time.time()
                    if now - self._last_version_check >= 10.0:
                        self._last_version_check = now
                        self._check_embedding_version()

                # 2. Refresh embeddings định kỳ dự phòng
                if self._server_online:
                    # Refresh global cache
                    if self.should_refresh_embeddings():
                        self.pull_embeddings()
                        
                    # Refresh multi-caches cho các camera đang kết nối
                    for cam_id in list(self._multi_caches.keys()):
                        if self.should_refresh_embeddings(cam_id):
                            self.pull_embeddings(cam_id)

            except Exception as e:
                logger.debug(f"Sync loop error: {e}")

            # Ngủ theo chu kỳ
            for _ in range(edge_config.sync_interval_sec):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

    def _check_embedding_version(self):
        """
        [FIX] Kiểm tra nhanh xem có học viên mới đăng ký không.
        Nếu có phân bản mới → pull lại toàn bộ embeddings — không cần restart Mini PC.
        """
        try:
            resp = self._session.get(
                f"{self.server_url}/api/embeddings/version",
                headers=self._headers(),
                timeout=2,
            )
            if resp.status_code == 200:
                data = resp.json()
                server_version = data.get("embedding_version", 0)
                if server_version != self._known_embedding_version:
                    logger.info(f"📥 Phát hiện embedding_version mới: {self._known_embedding_version} → {server_version} — Đang pull...") 
                    self.pull_embeddings()
                    for cam_id in list(self._multi_caches.keys()):
                        self.pull_embeddings(cam_id)
                    self._known_embedding_version = server_version
        except Exception as e:
            logger.debug(f"_check_embedding_version error: {e}")

    def _discovery_loop(self):
        """Vòng lặp nền quét mạng LAN tìm IP Camera."""
        time.sleep(2) # Chờ app khởi động
        while not self._stop_event.is_set():
            try:
                cam_details = discover_network_cameras(timeout=3.0)
                links = generate_rtsp_links(cam_details)
                if links:
                    self._discovered_rtsp = links
            except Exception as e:
                logger.debug(f"Discovery error: {e}")
            
            # Quét định kỳ mỗi 60 giây
            for _ in range(60):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

    def _push_offline_records(self):
        """Đẩy các bản ghi offline lên Server."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT id, camera_id, timestamp, embedding, liveness_score, liveness_checked "
                    "FROM offline_attendance WHERE synced = 0 ORDER BY id"
                ).fetchall()

                if not rows:
                    return

                logger.info(f"🔄 Đồng bộ {len(rows)} bản ghi offline...")

                for row in rows:
                    rid, cam_id, ts, emb_bytes, ls, lc = row
                    embedding = np.frombuffer(emb_bytes, dtype=np.float32)

                    payload = {
                        "camera_id": cam_id,
                        "timestamp": ts,
                        "embedding": embedding.tolist(),
                        "liveness_score": ls,
                        "liveness_checked": bool(lc),
                    }

                    try:
                        resp = requests.post(
                            f"{self.server_url}/api/attendance",
                            json=payload,
                            headers=self._headers(),
                            timeout=5,
                        )
                        if resp.status_code == 200:
                            conn.execute(
                                "UPDATE offline_attendance SET synced = 1 WHERE id = ?",
                                (rid,),
                            )
                            conn.commit()
                            logger.success(f"✅ Đồng bộ offline record #{rid} thành công")
                        else:
                            logger.warning(f"Server từ chối record #{rid}: {resp.text}")
                            break
                    except Exception:
                        logger.warning("Mất kết nối khi sync — dừng batch")
                        break

        except Exception as e:
            logger.error(f"Lỗi push offline: {e}")

    def get_offline_count(self) -> int:
        """Đếm số bản ghi chờ đồng bộ."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM offline_attendance WHERE synced = 0"
                ).fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    def get_system_command(self) -> str:
        """Lấy lệnh hệ thống từ Server (START/STOP)."""
        data = self.get_system_command_raw()
        return data.get("command", "STOP")

    def get_system_command_raw(self) -> dict:
        """Lấy toàn bộ dữ liệu lệnh từ Server (bao gồm cả target_camera)."""
        try:
            resp = requests.get(
                f"{self.server_url}/api/system/command",
                headers=self._headers(),
                timeout=2,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return {"command": "STOP", "target_camera": None}

    def stop(self):
        """Dừng background sync."""
        self._stop_event.set()
        if self._sync_thread.is_alive():
            self._sync_thread.join(timeout=3)
        logger.info("EdgeClient đã dừng")


# Singleton
edge_client = EdgeClient()