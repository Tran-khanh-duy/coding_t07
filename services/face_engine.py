"""
services/face_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trái tim của hệ thống — Pipeline nhận dạng khuôn mặt

Luồng xử lý 1 frame:
  Frame 720p
    → RetinaFace: Detect khuôn mặt + 5 landmarks
    → Crop + Align: Chuẩn hoá về 112×112
    → ArcFace: Trích xuất vector 512 chiều
    → Cosine Similarity: So sánh với cache RAM
    → Kết quả: (student_id, name, score) trong ≤ 117ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
    logger.warning("InsightFace chưa được cài. Chạy: pip install insightface")

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
        return f"Không xác định ({self.similarity*100:.1f}%)"

    @property
    def box_color(self) -> tuple:
        """Màu bounding box: xanh lá = nhận ra, đỏ = không nhận ra."""
        return (0, 220, 100) if self.recognized else (60, 60, 220)


# ─────────────────────────────────────────────
#  FaceEngine — Singleton
# ─────────────────────────────────────────────

class FaceEngine:
    """
    Engine nhận dạng khuôn mặt — Singleton, thread-safe.

    Sử dụng:
        engine = FaceEngine.get_instance()
        results = engine.process_frame(frame, cache)
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
        self._load_lock = threading.Lock()
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
        """
        Load buffalo_l model lên GPU.
        Gọi 1 lần khi khởi động app — không gọi trong vòng lặp!
        Thời gian load: ~3-5 giây lần đầu.
        """
        if self._model_loaded and not force_reload:
            return True

        if not INSIGHTFACE_AVAILABLE:
            logger.error("InsightFace chưa cài đặt!")
            return False

        with self._load_lock:
            if self._model_loaded and not force_reload:
                return True

            try:
                logger.info(f"Đang load model '{ai_config.model_name}'...")
                t0 = time.perf_counter()

                self._app = FaceAnalysis(
                    name=ai_config.model_name,
                    root=str(ai_config.model_pack_dir),
                    providers=ai_config.onnx_providers,   # từ config — mặc định CPU
                )
                # ctx_id=-1 → CPU (940MX Compute 5.0 không đủ cho một số CUDA kernel)
                # ctx_id=0  → GPU (cần Compute >= 6.0)
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
        """
        Phát hiện tất cả khuôn mặt trong frame.
        Input: BGR frame từ OpenCV
        Output: danh sách DetectedFace
        """
        if not self.is_ready:
            logger.warning("Model chưa được load!")
            return []

        try:
            # InsightFace nhận BGR (giống OpenCV)
            faces = self._app.get(frame)
            result = []
            for f in faces:
                # Lọc theo ngưỡng độ tin cậy detection
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

    # ─── Nhận diện khuôn mặt ─────────────────

    def recognize(
        self,
        face: DetectedFace,
        cache: EmbeddingCache,
    ) -> RecognitionResult:
        """
        So sánh embedding của face với toàn bộ embeddings trong cache.
        Dùng Cosine Similarity vectorized (numpy) — cực nhanh.
        """
        base = RecognitionResult(
            bbox=face.bbox, det_score=face.det_score,
            recognized=False, student_id=None, student_code=None,
            full_name=None, class_id=None, similarity=0.0,
        )

        if face.embedding is None or cache.is_empty:
            return base

        # Chuẩn hoá L2 embedding đầu vào
        emb = face.embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm < 1e-8:
            return base
        emb = emb / norm

        # ── Cosine Similarity (dot product sau khi normalize) ──
        # cache.embeddings: matrix (N, 512) — đã normalize sẵn
        # emb: vector (512,)
        # scores: vector (N,) — giá trị từ -1 đến 1
        scores = cache.embeddings @ emb          # shape: (N,)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= ai_config.recognition_threshold:
            # class_ids có thể rỗng nếu cache được tạo không đầy đủ
            cid = cache.class_ids[best_idx] if cache.class_ids else None
            return RecognitionResult(
                bbox=face.bbox,
                det_score=face.det_score,
                recognized=True,
                student_id=cache.student_ids[best_idx],
                student_code=cache.student_codes[best_idx],
                full_name=cache.full_names[best_idx],
                class_id=cid,
                similarity=best_score,
            )
        else:
            base.similarity = best_score
            return base

    # ─── Xử lý toàn bộ 1 frame ───────────────

    def process_frame(
        self,
        frame: np.ndarray,
        cache: EmbeddingCache,
    ) -> tuple[list[RecognitionResult], float]:
        """
        Xử lý 1 frame hoàn chỉnh:
          1. Detect tất cả khuôn mặt
          2. Nhận diện từng khuôn mặt
          3. Trả về kết quả + thời gian xử lý (ms)

        Returns:
            (results, elapsed_ms)
        """
        t0 = time.perf_counter()

        detected = self.detect_faces(frame)
        results = []

        for face in detected:
            t_rec = time.perf_counter()
            result = self.recognize(face, cache)
            result.process_time_ms = (time.perf_counter() - t_rec) * 1000
            results.append(result)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Cập nhật thống kê
        self._stats["total_frames"] += 1
        self._stats["total_faces"]  += len(results)
        self._stats["recognized"]   += sum(1 for r in results if r.recognized)
        self._stats["last_time_ms"]  = elapsed_ms
        # Moving average
        n = self._stats["total_frames"]
        self._stats["avg_time_ms"] = (
            self._stats["avg_time_ms"] * (n - 1) + elapsed_ms
        ) / n

        return results, elapsed_ms

    # ─── Enroll: tính embedding cho học viên mới ─

    def compute_enrollment_embedding(
        self,
        photos: list[np.ndarray],
    ) -> tuple[Optional[np.ndarray], float, int]:
        """
        Tính embedding đại diện cho học viên mới từ nhiều ảnh.
        Dùng mean pooling + L2 normalize.

        Args:
            photos: Danh sách ảnh BGR (từ camera)

        Returns:
            (embedding, avg_det_score, valid_count)
            embedding = None nếu không có ảnh hợp lệ
        """
        embeddings = []
        det_scores = []

        for i, photo in enumerate(photos):
            faces = self.detect_faces(photo)
            if not faces:
                logger.debug(f"Ảnh {i+1}: Không phát hiện khuôn mặt")
                continue
            if len(faces) > 1:
                logger.debug(f"Ảnh {i+1}: Phát hiện {len(faces)} khuôn mặt — bỏ qua")
                continue

            face = faces[0]
            if face.embedding is not None:
                embeddings.append(face.embedding)
                det_scores.append(face.det_score)
                logger.debug(f"Ảnh {i+1}: OK (det_score={face.det_score:.3f})")

        if not embeddings:
            return None, 0.0, 0

        # Mean pooling → normalize L2
        mean_emb = np.mean(embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-8:
            mean_emb = mean_emb / norm

        avg_score = float(np.mean(det_scores))
        logger.info(f"Enrollment: {len(embeddings)}/{len(photos)} ảnh hợp lệ, avg_score={avg_score:.3f}")
        return mean_emb, avg_score, len(embeddings)

    # ─── Helper methods cho test / external use ──

    def get_embedding(self, face_region: np.ndarray) -> "np.ndarray | None":
        """
        Lấy embedding vector (512D) từ 1 vùng ảnh mặt đã crop.
        face_region: ảnh BGR bất kỳ kích thước (sẽ resize về 112×112).
        Trả về numpy array (512,) đã normalize L2, hoặc None nếu không detect được.
        """
        if not self.is_ready:
            return None
        try:
            # Resize về 112×112 — kích thước chuẩn của ArcFace
            face_img = cv2.resize(face_region, (112, 112))
            if face_img.ndim == 2:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)

            # Thử detect trên frame (có thể không tìm được mặt trong ảnh crop nhỏ)
            faces = self.detect_faces(face_img)
            if faces and faces[0].embedding is not None:
                return faces[0].embedding

            # Fallback: dùng recognition model trực tiếp nếu app có sẵn
            if hasattr(self._app, "models") and self._app.models:
                for model in self._app.models:
                    if hasattr(model, "get_feat"):
                        feat = model.get_feat([face_img])
                        if feat is not None and len(feat) > 0:
                            emb = feat[0].astype(np.float32)
                            norm = np.linalg.norm(emb)
                            if norm > 1e-8:
                                return emb / norm
            return None
        except Exception as e:
            logger.debug(f"get_embedding error: {e}")
            return None

    def find_match(
        self,
        embedding: np.ndarray,
        cache: "EmbeddingCache",
    ) -> "RecognitionResult | None":
        """
        Tìm học viên khớp nhất với embedding trong cache.
        Wrapper gọn cho test / external use.
        Trả về RecognitionResult (recognized=True/False), hoặc None nếu cache rỗng.
        """
        if cache is None or cache.is_empty:
            return None

        # Tạo DetectedFace giả để tái dùng recognize()
        dummy_face = DetectedFace(
            bbox=np.array([0, 0, 112, 112]),
            det_score=1.0,
            embedding=embedding,
            landmarks=np.zeros((5, 2), dtype=np.float32),
        )
        return self.recognize(dummy_face, cache)

    # ─── Vẽ kết quả lên frame ────────────────

    def draw_results(
        self,
        frame: np.ndarray,
        results: list[RecognitionResult],
        elapsed_ms: float = 0.0,
    ) -> np.ndarray:
        """
        Vẽ bounding box + tên + thông tin lên frame.
        Trả về frame mới (không sửa frame gốc).
        """
        output = frame.copy()

        for res in results:
            x1, y1, x2, y2 = res.bbox
            color = res.box_color

            # Bounding box với góc bo tròn (vẽ thủ công)
            thickness = 2
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            # Nhãn tên
            label = res.display_name
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            (lw, lh), _ = cv2.getTextSize(label, font, font_scale, 2)

            # Nền nhãn
            pad = 4
            cv2.rectangle(
                output,
                (x1, y1 - lh - pad * 2 - 2),
                (x1 + lw + pad * 2, y1),
                color, -1
            )
            # Chữ
            cv2.putText(
                output, label,
                (x1 + pad, y1 - pad - 2),
                font, font_scale, (255, 255, 255), 2, cv2.LINE_AA
            )

        # Thông tin FPS / latency ở góc trên trái
        info = f"{elapsed_ms:.0f}ms | {len(results)} face(s)"
        cv2.putText(
            output, info,
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 220, 255), 2, cv2.LINE_AA
        )

        return output

    # ─── Thống kê ─────────────────────────────

    def get_stats(self) -> dict:
        s = self._stats.copy()
        if s["total_faces"] > 0:
            s["recognition_rate"] = s["recognized"] / s["total_faces"] * 100
        else:
            s["recognition_rate"] = 0.0
        return s

    def reset_stats(self):
        for k in self._stats:
            self._stats[k] = 0 if isinstance(self._stats[k], int) else 0.0


# ─────────────────────────────────────────────
#  Singleton instance
# ─────────────────────────────────────────────
face_engine = FaceEngine.get_instance()