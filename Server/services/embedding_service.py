import threading
import time
import numpy as np
try:
    import faiss
except ImportError:
    faiss = None
    
from loguru import logger
from typing import Optional, Tuple

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from database.repositories import embedding_repo
from core.state_manager import state_manager
from database.models import EmbeddingCache

class EmbeddingManager:
    """
    Service chuyên trách xử lý Nhận diện khuôn mặt (Embeddings).
    - Quản lý FAISS index (Hoặc Fallback sang Numpy Matrix)
    - Lazy loading từ Database
    - Thread-safe cosine search
    """
    _instance: Optional["EmbeddingManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        self._rlock = threading.RLock()
        
        self.faiss_index = None
        self.cache = EmbeddingCache()
        self._last_reload = 0.0
        self._embedding_version = 0
        self._initialized = True

    def load(self) -> bool:
        """Tải toàn bộ Embeddings từ DB và build FAISS Index."""
        logger.info("Đang build Embedding Index từ Database...")
        t0 = time.perf_counter()
        
        try:
            new_cache = embedding_repo.load_all_to_cache()
            
            with self._rlock:
                self.cache = new_cache
                
                if not new_cache.is_empty:
                    # Chẩn hóa vector (Cosine Similarity -> Inner Product)
                    norms = np.linalg.norm(new_cache.embeddings, axis=1, keepdims=True)
                    norms[norms == 0] = 1e-8
                    normalized_embeddings = new_cache.embeddings / norms
                    
                    if faiss is not None:
                        # Khởi tạo FAISS IndexFlatIP (Inner Product)
                        d = normalized_embeddings.shape[1]
                        self.faiss_index = faiss.IndexFlatIP(d)
                        self.faiss_index.add(normalized_embeddings.astype(np.float32))
                    else:
                        # Fallback sang Numpy Matrix lưu trực tiếp
                        self.faiss_index = normalized_embeddings.astype(np.float32)
                else:
                    self.faiss_index = None

                self._last_reload = time.time()
                # Tăng version trên Redis
                self._embedding_version, _ = state_manager.increment_embedding_version()

            elapsed = (time.perf_counter() - t0) * 1000
            if new_cache.is_empty:
                logger.warning("Embedding Index: Database trống!")
            else:
                engine = "FAISS" if faiss is not None else "NUMPY"
                logger.success(f"✅ {engine} Index build thành công: {new_cache.size} vectors | {elapsed:.1f}ms")
            
            return True
        except Exception as e:
            logger.error(f"Lỗi build Index: {e}")
            return False

    def reload(self):
        """Force reload."""
        self.load()

    def search(self, incoming_vector: np.ndarray, top_k: int = 1, valid_ids: list[int] = None) -> Tuple[float, int]:
        """
        Tìm kiếm khuôn mặt bằng FAISS / Numpy.
        Trả về (Best Score, Real Index). Trạng thái rỗng -> return (-1.0, -1)
        """
        with self._rlock:
            if self.faiss_index is None or self.cache.is_empty:
                return -1.0, -1

            # Chuẩn hóa vector đầu vào
            emb = incoming_vector.astype(np.float32).flatten()
            norm = np.linalg.norm(emb)
            if norm < 1e-8:
                return -1.0, -1
            emb = (emb / norm).reshape(1, -1)

            if faiss is not None:
                # FAISS Mạch tìm kiếm
                k = min(self.cache.size, top_k if valid_ids is None else self.cache.size)
                distances, indices = self.faiss_index.search(emb, k)
                
                if valid_ids is not None:
                    # Lọc thủ công FAISS results theo valid_ids
                    for idx_rank in range(k):
                        real_idx = indices[0][idx_rank]
                        if self.cache.student_ids[real_idx] in valid_ids:
                            return float(distances[0][idx_rank]), real_idx
                    return -1.0, -1

                return float(distances[0][0]), indices[0][0]
            else:
                # Numpy Mạch tìm kiếm
                similarities = self.faiss_index @ emb.T
                similarities = similarities.flatten()
                
                if valid_ids is not None:
                    for i in range(self.cache.size):
                        if self.cache.student_ids[i] not in valid_ids:
                            similarities[i] = -1.0
                
                best_idx = int(np.argmax(similarities))
                best_score = float(similarities[best_idx])
                return best_score, best_idx
            
    def get_student_info(self, idx: int) -> dict:
        """Lấy thông tin học viên dựa vào index."""
        with self._rlock:
            if idx < 0 or idx >= self.cache.size:
                return {}
            return {
                "student_id": self.cache.student_ids[idx],
                "student_code": self.cache.student_codes[idx],
                "full_name": self.cache.full_names[idx],
                "class_name": self.cache.class_names[idx],
                "class_id": self.cache.class_ids[idx],
            }

    def get_all_embeddings(self) -> dict:
        """Trả về toàn bộ embeddings (dùng cho API sync xuống Mini PC)."""
        with self._rlock:
            return {
                "student_ids": self.cache.student_ids,
                "student_codes": self.cache.student_codes,
                "full_names": self.cache.full_names,
                "class_names": self.cache.class_names,
                "class_ids": self.cache.class_ids,
                "embeddings": self.cache.embeddings # Mảng numpy gốc
            }

    @property
    def size(self):
        with self._rlock:
            return self.cache.size

    @property
    def version(self):
        return self._embedding_version

embedding_service = EmbeddingManager()
