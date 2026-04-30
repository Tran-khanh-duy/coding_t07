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
except Exception as e:
    INSIGHTFACE_AVAILABLE = False
    logger.error(f"⚠️ Lỗi khởi động InsightFace (Có thể do Driver hoặc DLL): {e}")
    logger.warning("Vui lòng cắm sạc Laptop và để chế độ Best Performance.")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ai_config, app_config, anti_spoof_config
from database.models import EmbeddingCache
# Anti-Spoofing Service (Có cơ chế dự phòng nếu Torch lỗi DLL)
try:
    if anti_spoof_config.enabled:
        from services.anti_spoof_service import anti_spoof_service
        if anti_spoof_service is not None:
            ANTI_SPOOF_AVAILABLE = True
            logger.success("🚀 Anti-Spoofing Module đã import thành công. Model sẽ được load lazily!")
        else:
            ANTI_SPOOF_AVAILABLE = False
            logger.warning("⚠️ Không thể khởi tạo Anti-Spoofing Service.")
    else:
        ANTI_SPOOF_AVAILABLE = False
        logger.info("ℹ️ Anti-Spoofing đang bị tắt trong cấu hình.")
except Exception as e:
    # Lỗi thường gặp trên Laptop: WinError 1114 do Torch DLL
    if anti_spoof_config.enabled:
        logger.error(f"⚠️ Không thể khởi động Anti-Spoofing (Lỗi hệ thống): {e}")
        logger.warning("Hệ thống sẽ chạy ở chế độ NHẬN DIỆN THƯỜNG (Tắt chống giả mạo).")
    ANTI_SPOOF_AVAILABLE = False


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
    class_name:     Optional[str]
    class_code:     Optional[str]
    similarity:     float           # Cosine similarity (0.0 - 1.0)
    
    # Anti-Spoofing
    is_real:        bool = True     # Mặc định là thật nếu không check hoặc lỗi
    spoof_score:    float = 1.0

    # Meta
    process_time_ms: float = 0.0

    @property
    def display_name(self) -> str:
        if self.recognized:
            cls = f" - {self.class_code}" if self.class_code else ""
            return f"{self.full_name}{cls} ({self.similarity*100:.1f}%)"
        return f"Khách / Lạ ({self.similarity*100:.1f}%)"

    @property
    def box_color(self) -> tuple:
        """Màu bounding box: xanh lá = nhận ra, đỏ = không nhận ra, cam = spoof."""
        if not self.is_real:
            return (0, 140, 255)  # Orange (BGR)
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

                try:
                    self._app = FaceAnalysis(
                        name=ai_config.model_name,
                        root=str(ai_config.model_pack_dir),
                        providers=ai_config.onnx_providers,
                    )
                    self._app.prepare(
                        ctx_id=ai_config.gpu_ctx_id,
                        det_size=ai_config.det_size,
                    )
                except Exception as e:
                    logger.warning(f"CUDA loading failed ({e}), falling back to CPU...")
                    self._app = FaceAnalysis(
                        name=ai_config.model_name,
                        root=str(ai_config.model_pack_dir),
                        providers=['CPUExecutionProvider'],
                    )
                    self._app.prepare(
                        ctx_id=-1, # Force CPU
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

    def unload_model(self):
        """Giải phóng model khỏi GPU/RAM để tránh tràn bộ nhớ."""
        with self._load_lock:
            if self._app:
                del self._app
                self._app = None
            self._model_loaded = False
            
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("🗑️ Đã giải phóng AI Model khỏi bộ nhớ (RAM/VRAM).")

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

        FIX: Dùng result_map theo chỉ số gốc để đảm bảo thứ tự OUTPUT luôn khớp
        với thứ tự INPUT faces — tránh lỗi nhãn tên bị gán nhầm bounding box.
        FIX: One-to-one assignment — mỗi học sinh chỉ được gán cho 1 khuôn mặt
        có score cao nhất, tránh trường hợp 2 khuôn mặt đều nhận ra cùng 1 người.
        """
        if not faces:
            return []

        # Nếu Cache trống (Chưa có học sinh nào trong DB)
        if cache is None or cache.is_empty:
            return [
                RecognitionResult(
                    bbox=face.bbox, det_score=face.det_score,
                    recognized=False, student_id=None, student_code=None,
                    full_name=None, class_id=None, class_name=None, class_code=None, similarity=0.0
                )
                for face in faces
            ]

        # ── Bước 1: Tách khuôn mặt hợp lệ, giữ nguyên index gốc ──────────────
        # result_map[i] = RecognitionResult cho faces[i]
        result_map: dict[int, RecognitionResult] = {}

        # (orig_index, face) của những mặt có embedding hợp lệ
        valid_pairs: list[tuple[int, DetectedFace]] = []
        embeddings_list: list[np.ndarray] = []

        for i, face in enumerate(faces):
            if face.embedding is not None:
                valid_pairs.append((i, face))
                embeddings_list.append(face.embedding)
            else:
                # Không có embedding → đánh dấu unknown ngay, giữ đúng vị trí
                result_map[i] = RecognitionResult(
                    bbox=face.bbox, det_score=face.det_score,
                    recognized=False, student_id=None, student_code=None,
                    full_name=None, class_id=None, similarity=0.0
                )

        if not embeddings_list:
            return [result_map[i] for i in range(len(faces))]

        # ── Bước 2: Ma trận cosine similarity (M, N) ──────────────────────────
        emb_matrix = np.array(embeddings_list, dtype=np.float32)

        # Chuẩn hoá L2 toàn bộ M khuôn mặt (vectorised)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1e-8
        emb_matrix = emb_matrix / norms

        # (M, 512) @ (512, N) → Ma trận điểm (M, N)
        scores_matrix = emb_matrix @ cache.embeddings.T  # shape (M, N)

        best_indices = np.argmax(scores_matrix, axis=1)   # (M,)
        best_scores  = np.max(scores_matrix,  axis=1)     # (M,)

        # ── Bước 3: One-to-one assignment — chống nhầm khi 2 mặt cùng match 1 người ──
        # Với mỗi student_id được chọn, chỉ khuôn mặt có score cao nhất được giữ;
        # những khuôn mặt còn lại của cùng student_id bị đánh dấu unknown.
        M = len(valid_pairs)
        assigned_face_for_student: dict[int, int] = {}  # student_cache_idx → face_local_idx

        for local_i in range(M):
            score = float(best_scores[local_i])
            if score < ai_config.recognition_threshold:
                continue  # Dưới ngưỡng, xử lý ở bước 4

            student_cache_idx = int(best_indices[local_i])
            if student_cache_idx not in assigned_face_for_student:
                # Học sinh này chưa được gán → gán cho khuôn mặt hiện tại
                assigned_face_for_student[student_cache_idx] = local_i
            else:
                # Học sinh này đã được gán → so sánh score, giữ score cao hơn
                prev_local_i = assigned_face_for_student[student_cache_idx]
                if score > float(best_scores[prev_local_i]):
                    assigned_face_for_student[student_cache_idx] = local_i

        # ── Bước 4: Xây dựng kết quả theo đúng thứ tự gốc ────────────────────
        # Tập hợp các local_i đã được "giành" bở one-to-one assignment
        winning_local_indices = set(assigned_face_for_student.values())

        for local_i, (orig_idx, face) in enumerate(valid_pairs):
            student_cache_idx = int(best_indices[local_i])
            best_score = float(best_scores[local_i])

            is_above_threshold = best_score >= ai_config.recognition_threshold
            is_winner = local_i in winning_local_indices

            if is_above_threshold and is_winner:
                cid = cache.class_ids[student_cache_idx] if cache.class_ids else None
                cname = cache.class_names[student_cache_idx] if cache.class_names else None
                ccode = cache.class_codes[student_cache_idx] if cache.class_codes else None
                result_map[orig_idx] = RecognitionResult(
                    bbox=face.bbox,
                    det_score=face.det_score,
                    recognized=True,
                    student_id=cache.student_ids[student_cache_idx],
                    student_code=cache.student_codes[student_cache_idx],
                    full_name=cache.full_names[student_cache_idx],
                    class_id=cid,
                    class_name=cname,
                    class_code=ccode,
                    similarity=best_score,
                )
            else:
                # Dưới ngưỡng, hoặc thua trong one-to-one assignment
                result_map[orig_idx] = RecognitionResult(
                    bbox=face.bbox,
                    det_score=face.det_score,
                    recognized=False,
                    student_id=None, student_code=None, full_name=None, class_id=None, class_name=None, class_code=None,
                    similarity=best_score,
                )

        # Trả về theo đúng thứ tự gốc 0..len(faces)-1
        return [result_map[i] for i in range(len(faces))]

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
        
        # 3. Check Anti-Spoofing cho từng khuôn mặt (Nếu có sẵn)
        for i, res in enumerate(results):
            if ANTI_SPOOF_AVAILABLE:
                try:
                    is_real, s_score = anti_spoof_service.is_real(frame, res.bbox)
                    res.is_real = is_real
                    res.spoof_score = s_score
                    if not is_real:
                        logger.warning(f"Phát hiện SPOOF! Score: {s_score:.3f}")
                except Exception as e:
                    logger.debug(f"Lỗi runtime Anti-Spoofing: {e}")
                    res.is_real = True 
            else:
                # Chế độ dự phòng: Mặc định là thật
                res.is_real = True
                res.spoof_score = 1.0

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
            if not faces:
                continue
            # Lấy khuôn mặt lớn nhất trong trường hợp khung hình dính người khác phía sau
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
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
        
        # 1. Vẽ bounding box bằng OpenCV cho nhanh
        for res in results:
            x1, y1, x2, y2 = res.bbox
            color = res.box_color
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
        # 2. Vẽ Text tiếng Việt bằng PIL
        try:
            from PIL import Image, ImageDraw, ImageFont
            import os
            
            # Convert BGR to RGB for PIL
            img_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # Ưu tiên load font Arial để nhận Tiếng Việt (có sẵn trên Windows)
            try:
                # Kích thước font chữ
                font = ImageFont.truetype("arial.ttf", 22)
            except:
                font = ImageFont.load_default()

            for res in results:
                x1, y1, x2, y2 = res.bbox
                label = res.display_name
                
                # Convert BGR (OpenCV) -> RGB (PIL) cho màu nền chữ
                b, g, r = res.box_color
                rgb_color = (r, g, b)
                
                # Tính kích thước chuỗi text
                bbox_text = draw.textbbox((0, 0), label, font=font)
                tw = bbox_text[2] - bbox_text[0]
                th = bbox_text[3] - bbox_text[1]
                pad = 5
                
                # Vẽ nền màu trùng màu box, ở phía trên bounding box
                draw.rectangle(
                    [x1, y1 - th - pad * 2, x1 + tw + pad * 2, y1],
                    fill=rgb_color
                )
                # Vẽ chữ trắng
                draw.text((x1 + pad, y1 - th - pad - 2), label, font=font, fill=(255, 255, 255))

            # Thông số góc trái trên
            info = f"{elapsed_ms:.0f}ms | {len(results)} face(s)"
            draw.text((12, 12), info, font=font, fill=(255, 200, 0)) # Màu vàng cam (RGB)

            # Chuyển ngược lại về BGR
            output = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
        except ImportError:
            # Fallback nếu máy không install PIL (dùng OpenCV cũ sẽ bị lỗi dấu hỏi)
            for res in results:
                x1, y1, x2, y2 = res.bbox
                color = res.box_color
                label = res.display_name.encode('ascii', 'ignore').decode('utf-8')
                font, font_scale = cv2.FONT_HERSHEY_SIMPLEX, 0.6
                (lw, lh), _ = cv2.getTextSize(label, font, font_scale, 2)
                pad = 4
                cv2.rectangle(output, (x1, y1 - lh - pad * 2 - 2), (x1 + lw + pad * 2, y1), color, -1)
                cv2.putText(output, label, (x1 + pad, y1 - pad - 2), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

            info = f"{elapsed_ms:.0f}ms | {len(results)} face(s)"
            cv2.putText(output, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
            
        except Exception as e:
            logger.error(f"Lỗi vẽ label bằng PIL: {e}")

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