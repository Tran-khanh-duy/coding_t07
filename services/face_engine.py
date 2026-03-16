"""
services/face_engine.py

Luồng xử lý 1 frame:
  Frame 720p/1080p
    → RetinaFace: Detect khuôn mặt + 5 landmarks (Thread-Safe)
    → Crop + Align: Chuẩn hoá về 112×112
    → ArcFace: Trích xuất vector 512 chiều
    → Batch Cosine Similarity: So sánh song song toàn bộ khuôn mặt bằng Ma Trận (Cực nhanh)
    → Kết quả: (student_id, name, score) trong vài milliseconds
"""
import time
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from loguru import logger

# InsightFace — bao gồm cả RetinaFace và ArcFace
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("InsightFace chưa được cài. Chạy: pip install insightface onnxruntime-gpu")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ai_config, app_config
from database.models import EmbeddingCache


# ─────────────────────────────────────────────
#  Data classes kết quả
# ─────────────────────────────────────────────

@dataclass
class DetectedFace:
    """1 khuôn mặt được phát hiện trong frame."""
    bbox:       np.ndarray      # [x1, y1, x2, y2]
    landmarks:  np.ndarray      # 5 điểm: mắt trái, mắt phải, mũi, miệng trái, miệng phải
    det_score:  float           # Độ tin cậy detection (0.0 - 1.0)
    embedding:  Optional[np.ndarray] = None   # Vector 512D sau khi qua ArcFace


@dataclass
class RecognitionResult:
    """Kết quả nhận diện cho 1 khuôn mặt."""
    # Thông tin khuôn mặt
    bbox:           np.ndarray
    det_score:      float

    # Kết quả nhận diện
    recognized:     bool            # True nếu nhận ra (score >= threshold)
    student_id:     Optional[int]   # None nếu không nhận ra
    student_code:   Optional[str]
    full_name:      Optional[str]
    class_id:       Optional[int]
    similarity:     float           # Cosine similarity (0.0 - 1.0)

    # Meta
    process_time_ms: float = 0.0

    @property
    def display_name(self) -> str:
        if self.recognized:
            return f"{self.full_name} ({self.similarity*100:.1f}%)"
        return f"Khach / La ({self.similarity*100:.1f}%)"

    @property
    def box_color(self) -> tuple:
        """Màu bounding box: xanh lá = nhận ra, đỏ = không nhận ra."""
        return (0, 220, 100) if self.recognized else (60, 60, 220)


# ─────────────────────────────────────────────
#  FaceEngine — Singleton
# ─────────────────────────────────────────────

class FaceEngine:
    """
    Engine nhận dạng khuôn mặt — Singleton, thread-safe cho Multi-Camera.

    Sử dụng:
        engine = FaceEngine.get_instance()
        results, ms = engine.process_frame(frame, cache)
    """
    _instance: Optional["FaceEngine"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> "FaceEngine":
        return cls()

    def __init__(self):
        if self._initialized:
            return

        self._app: Optional[FaceAnalysis] = None
        self._model_loaded = False
        
        # Locks
        self._load_lock = threading.Lock()
        
        # NÂNG CẤP 1: Inference Lock bảo vệ GPU khỏi xung đột khi hàng chục Camera gọi vào cùng 1 thời điểm
        self._inference_lock = threading.Lock() 
        
        self._initialized = True

        # Thống kê hiệu suất
        self._stats = {
            "total_frames":     0,
            "total_faces":      0,
            "recognized":       0,
            "avg_time_ms":      0.0,
            "last_time_ms":     0.0,
        }

    # ─── Load model ───────────────────────────

    def load_model(self, force_reload: bool = False) -> bool:
        """Load buffalo_l model lên GPU."""
        if self._model_loaded and not force_reload:
            return True

        if not INSIGHTFACE_AVAILABLE:
            logger.error("InsightFace chưa cài đặt!")
            return False

        with self._load_lock:
            if self._model_loaded and not force_reload:
                return True

            try:
                logger.info(f"Đang load model GPU '{ai_config.model_name}'...")
                t0 = time.perf_counter()

                self._app = FaceAnalysis(
                    name=ai_config.model_name,
                    root=str(ai_config.model_pack_dir),
                    providers=ai_config.onnx_providers,
                )
                
                self._app.prepare(
                    ctx_id=ai_config.gpu_ctx_id,
                    det_size=ai_config.det_size,
                )

                elapsed = (time.perf_counter() - t0) * 1000
                self._model_loaded = True

                # Xác định đang dùng GPU hay CPU
                import onnxruntime as ort
                device = ort.get_device()
                logger.success(
                    f"✅ Model '{ai_config.model_name}' đã load xong! "
                    f"Device: {device} | Thời gian: {elapsed:.0f}ms"
                )
                return True

            except Exception as e:
                logger.error(f"Lỗi load model: {e}")
                self._model_loaded = False
                return False

    @property
    def is_ready(self) -> bool:
        return self._model_loaded and self._app is not None

    # ─── Detect khuôn mặt ─────────────────────

    def detect_faces(self, frame: np.ndarray) -> list[DetectedFace]:
        """Phát hiện tất cả khuôn mặt trong frame (Thread-Safe)."""
        if not self.is_ready:
            return []

        try:
            # Khóa luồng khi quét qua Neural Network để tránh crash vRAM
            with self._inference_lock:
                faces = self._app.get(frame)
                
            result = []
            for f in faces:
                if f.det_score < ai_config.min_face_det_score:
                    continue
                result.append(DetectedFace(
                    bbox=f.bbox.astype(int),
                    landmarks=f.kps,
                    det_score=float(f.det_score),
                    embedding=f.embedding,   # ArcFace embedding 512D
                ))
            return result

        except Exception as e:
            logger.error(f"Lỗi detect_faces: {e}")
            return []


    # ─── NÂNG CẤP 2: Nhận diện HÀNG LOẠT (Batch Processing) ───

    def recognize_batch(
        self,
        faces: list[DetectedFace],
        cache: EmbeddingCache,
    ) -> list[RecognitionResult]:
        """
        Nhận diện TẤT CẢ khuôn mặt trong frame cùng một lúc bằng Phép nhân Ma Trận.
        10 người trong khung hình sẽ được xử lý chung trong 1 phép toán duy nhất.
        """
        if not faces:
            return []

        results = []
        
        # Nếu Cache trống (Chưa có học sinh nào trong DB)
        if cache is None or cache.is_empty:
            for face in faces:
                results.append(RecognitionResult(
                    bbox=face.bbox, det_score=face.det_score,
                    recognized=False, student_id=None, student_code=None,
                    full_name=None, class_id=None, similarity=0.0
                ))
            return results

        # 1. Trích xuất tất cả embeddings hợp lệ
        valid_faces = []
        embeddings_list = []
        
        for face in faces:
            if face.embedding is not None:
                valid_faces.append(face)
                embeddings_list.append(face.embedding)
            else:
                results.append(RecognitionResult(
                    bbox=face.bbox, det_score=face.det_score,
                    recognized=False, student_id=None, student_code=None,
                    full_name=None, class_id=None, similarity=0.0
                ))

        if not embeddings_list:
            return results

        # 2. Xử lý Ma Trận Thần Tốc bằng Numpy
        # emb_matrix có shape: (M, 512) với M là số khuôn mặt trong khung hình
        emb_matrix = np.array(embeddings_list, dtype=np.float32)
        
        # NÂNG CẤP 3: Chuẩn hoá L2 cho toàn bộ M khuôn mặt (Vectorized)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1e-8 # Tránh lỗi chia cho 0
        emb_matrix = emb_matrix / norms

        # Phép nhân ma trận: (M, 512) @ (512, N) -> Kết quả là Ma trận Điểm (M, N)
        # N là tổng số học sinh trong Database
        scores_matrix = emb_matrix @ cache.embeddings.T

        # Tìm học sinh có điểm cao nhất cho MỖI khuôn mặt
        best_indices = np.argmax(scores_matrix, axis=1)
        best_scores = np.max(scores_matrix, axis=1)

        # 3. Áp xạ lại kết quả
        for i, face in enumerate(valid_faces):
            best_idx = int(best_indices[i])
            best_score = float(best_scores[i])

            if best_score >= ai_config.recognition_threshold:
                cid = cache.class_ids[best_idx] if cache.class_ids else None
                results.append(RecognitionResult(
                    bbox=face.bbox,
                    det_score=face.det_score,
                    recognized=True,
                    student_id=cache.student_ids[best_idx],
                    student_code=cache.student_codes[best_idx],
                    full_name=cache.full_names[best_idx],
                    class_id=cid,
                    similarity=best_score,
                ))
            else:
                results.append(RecognitionResult(
                    bbox=face.bbox,
                    det_score=face.det_score,
                    recognized=False,
                    student_id=None, student_code=None, full_name=None, class_id=None,
                    similarity=best_score,
                ))

        return results

    # ─── Xử lý toàn bộ 1 frame ───────────────

    def process_frame(
        self,
        frame: np.ndarray,
        cache: EmbeddingCache,
    ) -> tuple[list[RecognitionResult], float]:
        t0 = time.perf_counter()

        # 1. Phát hiện tất cả khuôn mặt
        detected = self.detect_faces(frame)

        # 2. Nhận diện song song toàn bộ
        t_rec_start = time.perf_counter()
        results = self.recognize_batch(detected, cache)
        
        # Cập nhật thông số thời gian xử lý cho từng khuôn mặt
        t_rec_elapsed = (time.perf_counter() - t_rec_start) * 1000
        for r in results:
            r.process_time_ms = t_rec_elapsed / len(results) if results else 0.0

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Cập nhật thống kê
        self._stats["total_frames"] += 1
        self._stats["total_faces"]  += len(results)
        self._stats["recognized"]   += sum(1 for r in results if r.recognized)
        self._stats["last_time_ms"]  = elapsed_ms
        
        # Moving average
        n = self._stats["total_frames"]
        self._stats["avg_time_ms"] = (self._stats["avg_time_ms"] * (n - 1) + elapsed_ms) / n

        return results, elapsed_ms

    # ─── Mọi hàm Enroll, Vẽ Box, Thống kê bên dưới được giữ nguyên chuẩn mực ───

    def recognize(self, face: DetectedFace, cache: EmbeddingCache) -> RecognitionResult:
        """Hàm nhận diện đơn lẻ (Giữ lại để tương thích ngược)"""
        return self.recognize_batch([face], cache)[0] if cache else None

    def compute_enrollment_embedding(self, photos: list[np.ndarray]) -> tuple[Optional[np.ndarray], float, int]:
        embeddings = []
        det_scores = []
        for i, photo in enumerate(photos):
            faces = self.detect_faces(photo)
            if not faces or len(faces) > 1:
                continue
            face = faces[0]
            if face.embedding is not None:
                embeddings.append(face.embedding)
                det_scores.append(face.det_score)

        if not embeddings: return None, 0.0, 0
        mean_emb = np.mean(embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-8: mean_emb = mean_emb / norm
        return mean_emb, float(np.mean(det_scores)), len(embeddings)

    def get_embedding(self, face_region: np.ndarray) -> "np.ndarray | None":
        if not self.is_ready: return None
        try:
            face_img = cv2.resize(face_region, (112, 112))
            if face_img.ndim == 2: face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
            faces = self.detect_faces(face_img)
            if faces and faces[0].embedding is not None: return faces[0].embedding
            
            if hasattr(self._app, "models") and self._app.models:
                for model in self._app.models:
                    if hasattr(model, "get_feat"):
                        with self._inference_lock:
                            feat = model.get_feat([face_img])
                        if feat is not None and len(feat) > 0:
                            emb = feat[0].astype(np.float32)
                            norm = np.linalg.norm(emb)
                            if norm > 1e-8: return emb / norm
            return None
        except Exception as e:
            logger.debug(f"get_embedding error: {e}")
            return None

    def find_match(self, embedding: np.ndarray, cache: "EmbeddingCache") -> "RecognitionResult | None":
        if cache is None or cache.is_empty: return None
        dummy_face = DetectedFace(
            bbox=np.array([0, 0, 112, 112]), det_score=1.0, embedding=embedding,
            landmarks=np.zeros((5, 2), dtype=np.float32),
        )
        return self.recognize(dummy_face, cache)

    def draw_results(self, frame: np.ndarray, results: list[RecognitionResult], elapsed_ms: float = 0.0) -> np.ndarray:
        output = frame.copy()
        for res in results:
            x1, y1, x2, y2 = res.bbox
            color = res.box_color
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            label = res.display_name
            font, font_scale = cv2.FONT_HERSHEY_SIMPLEX, 0.6
            (lw, lh), _ = cv2.getTextSize(label, font, font_scale, 2)
            pad = 4
            cv2.rectangle(output, (x1, y1 - lh - pad * 2 - 2), (x1 + lw + pad * 2, y1), color, -1)
            cv2.putText(output, label, (x1 + pad, y1 - pad - 2), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        info = f"{elapsed_ms:.0f}ms | {len(results)} face(s)"
        cv2.putText(output, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
        return output

    def get_stats(self) -> dict:
        s = self._stats.copy()
        s["recognition_rate"] = (s["recognized"] / s["total_faces"] * 100) if s["total_faces"] > 0 else 0.0
        return s

    def reset_stats(self):
        for k in self._stats: self._stats[k] = 0 if isinstance(self._stats[k], int) else 0.0

# ─────────────────────────────────────────────
#  Singleton instance
# ─────────────────────────────────────────────
face_engine = FaceEngine.get_instance()