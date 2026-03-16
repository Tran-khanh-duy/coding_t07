"""
services/embedding_cache_manager.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Quản lý cache embeddings trong RAM.

Tại sao cần cache?
  - 1000 học viên × 512 float32 = ~2MB → hoàn toàn vừa trong RAM
  - Tìm kiếm trên RAM (numpy matrix): ~0.2ms
  - Tìm kiếm trên DB mỗi lần: ~10-50ms
  → Cache nhanh hơn 50-250 lần!

Hot reload:
  - Khi thêm học viên mới → reload cache không cần restart app
  - Dùng RLock để thread-safe trong khi reload
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import threading
import time
from typing import Optional
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.models import EmbeddingCache
from database.repositories import embedding_repo


class EmbeddingCacheManager:
    """
    Singleton quản lý cache embeddings.
    Thread-safe: nhiều camera thread cùng đọc cùng lúc an toàn.
    """
    _instance: Optional["EmbeddingCacheManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        # RLock cho phép cùng thread acquire nhiều lần (tránh deadlock)
        self._rlock = threading.RLock()
        self._cache: EmbeddingCache = EmbeddingCache()
        self._last_reload: float = 0.0
        self._reload_count: int = 0
        self._initialized = True

    # ─── Load / Reload ────────────────────────

    def load(self) -> bool:
        """
        Load toàn bộ embeddings từ DB vào RAM.
        Gọi 1 lần khi khởi động, hoặc sau khi thêm học viên mới.
        """
        logger.info("Đang load embeddings từ DB vào RAM cache...")
        t0 = time.perf_counter()

        try:
            new_cache = embedding_repo.load_all_to_cache()
            elapsed = (time.perf_counter() - t0) * 1000

            with self._rlock:
                self._cache = new_cache
                self._last_reload = time.time()
                self._reload_count += 1

            if new_cache.is_empty:
                logger.warning(
                    "Cache rỗng! Chưa có học viên nào được đăng ký khuôn mặt.\n"
                    "→ Vào tab 'Đăng ký học viên' để thêm học viên."
                )
            else:
                logger.success(
                    f"✅ Cache loaded: {new_cache.size} học viên | "
                    f"Matrix shape: {new_cache.embeddings.shape} | "
                    f"Thời gian: {elapsed:.1f}ms"
                )
            return True

        except Exception as e:
            logger.error(f"Lỗi load cache: {e}")
            return False

    def reload_after_enrollment(self, student_id: int = None):
        """
        Hot reload sau khi đăng ký học viên mới.
        Không restart app, không dừng camera.
        """
        name = f"student_id={student_id}" if student_id else "all"
        logger.info(f"Hot reload cache sau khi thêm {name}...")
        self.load()

    # ─── Đọc cache (thread-safe) ──────────────

    def get_cache(self) -> EmbeddingCache:
        """Lấy cache hiện tại — safe để dùng trong nhiều thread."""
        with self._rlock:
            return self._cache

    @property
    def size(self) -> int:
        with self._rlock:
            return self._cache.size

    @property
    def is_empty(self) -> bool:
        with self._rlock:
            return self._cache.is_empty

    # ─── Thêm học viên vào cache ngay lập tức ─

    def add_student_to_cache(self, student_id: int, student_code: str,
                              full_name: str, class_id: int,
                              embedding) -> bool:
        """
        Thêm 1 học viên vào cache mà không cần reload toàn bộ.
        Nhanh hơn reload() khi chỉ thêm 1 người.
        """
        import numpy as np
        try:
            emb = embedding.astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-8:
                emb = emb / norm

            with self._rlock:
                self._cache.student_ids.append(student_id)
                self._cache.student_codes.append(student_code)
                self._cache.full_names.append(full_name)
                self._cache.class_ids.append(class_id)

                if self._cache.embeddings is None:
                    self._cache.embeddings = emb.reshape(1, -1)
                else:
                    self._cache.embeddings = np.vstack([
                        self._cache.embeddings, emb
                    ])

            logger.info(f"Thêm vào cache: [{student_code}] {full_name} | Cache size: {self.size}")
            return True
        except Exception as e:
            logger.error(f"Lỗi add_student_to_cache: {e}")
            return False

    # ─── Thông tin debug ──────────────────────

    def get_info(self) -> dict:
        import datetime
        with self._rlock:
            return {
                "size":         self._cache.size,
                "shape":        str(self._cache.embeddings.shape) if self._cache.embeddings is not None else "None",
                "last_reload":  datetime.datetime.fromtimestamp(self._last_reload).strftime("%H:%M:%S") if self._last_reload else "Chưa load",
                "reload_count": self._reload_count,
                "memory_mb":    round(self._cache.embeddings.nbytes / 1024 / 1024, 3) if self._cache.embeddings is not None else 0,
            }


# Singleton instance
cache_manager = EmbeddingCacheManager()
