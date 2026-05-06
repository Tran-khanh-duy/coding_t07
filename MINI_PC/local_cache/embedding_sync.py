import os
import pickle
import threading
from loguru import logger
from pathlib import Path

CACHE_DIR = Path(__file__).parent
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class EmbeddingSyncManager:
    """
    Quản lý Đồng bộ và Lưu trữ Offline cho Embeddings (.pkl)
    Đảm bảo Mini PC vẫn nhận diện được khuôn mặt khi khởi động không có mạng (Offline-First).
    """
    def __init__(self):
        self._lock = threading.Lock()

    def _get_cache_path(self, camera_id: str) -> Path:
        safe_name = "".join([c for c in str(camera_id) if c.isalnum() or c in ('_', '-')]).rstrip()
        if not safe_name: safe_name = "default"
        return CACHE_DIR / f"embeddings_{safe_name}.pkl"

    def save_cache(self, camera_id: str, cache_obj, version: int):
        """Lưu EmbeddingCache xuống đĩa cứng (.pkl)"""
        path = self._get_cache_path(camera_id)
        with self._lock:
            try:
                data = {
                    "version": version,
                    "cache": cache_obj
                }
                with open(path, "wb") as f:
                    pickle.dump(data, f)
                logger.info(f"💾 Đã lưu Offline Embeddings (Version {version}) cho Camera {camera_id}")
            except Exception as e:
                logger.error(f"Lỗi lưu pkl cache {camera_id}: {e}")

    def load_cache(self, camera_id: str):
        """Tải EmbeddingCache từ đĩa cứng. Trả về (CacheObject, Version)"""
        path = self._get_cache_path(camera_id)
        with self._lock:
            if not path.exists():
                return None, 0
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                version = data.get("version", 0)
                cache_obj = data.get("cache")
                logger.success(f"📂 Đã tải Offline Embeddings (Version {version}) cho Camera {camera_id} từ đĩa.")
                return cache_obj, version
            except Exception as e:
                logger.error(f"Lỗi đọc pkl cache {camera_id}: {e}")
                return None, 0

embedding_sync = EmbeddingSyncManager()
