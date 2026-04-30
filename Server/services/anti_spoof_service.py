import os
import sys
import cv2
import numpy as np
from loguru import logger
from pathlib import Path
from config import anti_spoof_config

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    if anti_spoof_config.enabled:
        logger.warning("Thư viện 'torch' không khả dụng. Tính năng Anti-Spoofing sẽ bị TẮT.")


# Thêm Silent-Face-Anti-Spoofing-master vào sys.path để import các module của nó
# Thử tìm ở Server folder hoặc MINI_PC folder
base_path = Path(__file__).parent.parent
ANTI_SPOOF_ROOT = base_path / "Silent-Face-Anti-Spoofing-master"
if not ANTI_SPOOF_ROOT.exists():
    # Fallback cho Mini PC (thư mục Silent nằm trong MINI_PC)
    # Dùng Path(__file__).resolve() để lấy đường dẫn tuyệt đối chính xác nhất
    current_file = Path(__file__).resolve()
    # Tìm folder cha chứa cả Server và MINI_PC
    project_root = current_file.parents[2] 
    ANTI_SPOOF_ROOT = project_root / "MINI_PC" / "Silent-Face-Anti-Spoofing-master"

if ANTI_SPOOF_ROOT.exists():
    logger.info(f"✅ Tìm thấy Anti-Spoofing tại: {ANTI_SPOOF_ROOT}")
    if str(ANTI_SPOOF_ROOT) not in sys.path:
        sys.path.insert(0, str(ANTI_SPOOF_ROOT))
else:
    logger.warning(f"⚠️ Không tìm thấy folder Silent-Face-Anti-Spoofing-master. Vui lòng kiểm tra lại cấu hình.")

try:
    if anti_spoof_config.enabled and TORCH_AVAILABLE:
        from src.anti_spoof_predict import AntiSpoofPredict
        from src.generate_patches import CropImage
        from src.utility import parse_model_name, get_kernel
        from src.anti_spoof_predict import MODEL_MAPPING
    else:
        AntiSpoofPredict = None
        CropImage = None
except ImportError as e:
    if anti_spoof_config.enabled:
        logger.error(f"❌ Không thể import module Anti-Spoofing. Hãy đảm bảo folder Silent-Face-Anti-Spoofing-master tồn tại. Lỗi: {e}")
    else:
        # Nếu đã chủ động tắt thì chỉ log debug hoặc bỏ qua
        logger.debug("Anti-Spoofing đã bị tắt theo cấu hình.")


class AntiSpoofService:
    """
    Service xử lý chống giả mạo khuôn mặt (Anti-Spoofing).
    Sử dụng các model từ Silent-Face-Anti-Spoofing.
    Đã được tối ưu để cache model, tránh load lại mỗi lần predict.
    """
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if not TORCH_AVAILABLE:
            self.models_cache = {}
            return

        self.device = torch.device(f"cuda:{anti_spoof_config.device_id}" 
                                  if torch.cuda.is_available() and anti_spoof_config.device_id >= 0 
                                  else "cpu")
        self.model_dir = Path(anti_spoof_config.model_dir)
        
        # Sửa lỗi đường dẫn model khi chạy trên Server
        if not self.model_dir.exists():
            # Thử tìm trong thư mục MINI_PC
            project_root = Path(__file__).resolve().parents[2]
            fallback_dir = project_root / "MINI_PC" / "Silent-Face-Anti-Spoofing-master" / "resources" / "anti_spoof_models"
            if fallback_dir.exists():
                self.model_dir = fallback_dir
                logger.info(f"📂 Đã tự động chuyển hướng Model Anti-Spoofing sang: {self.model_dir}")
        
        self.model_dir = str(self.model_dir)
        
        # Khởi tạo predictor và cropper (Chỉ nếu module được import thành công)
        if AntiSpoofPredict is not None:
            self.predictor = AntiSpoofPredict(anti_spoof_config.device_id if anti_spoof_config.device_id >= 0 else 0)
        else:
            self.predictor = None
            
        if CropImage is not None:
            self.image_cropper = CropImage()
        else:
            self.image_cropper = None
        
        # Cache các model để tránh load lại
        self.models_cache = {}
        self._models_loaded = False

    def load_models(self):
        if self._models_loaded:
            return
        self._load_all_models()
        self._models_loaded = True

    def _load_all_models(self):
        if not TORCH_AVAILABLE or not os.path.exists(self.model_dir):
            logger.warning(f"Chức năng chống giả mạo (Anti-Spoofing) bị TẮT.")
            return

        if not os.path.isdir(self.model_dir):
             logger.warning(f"Đường dẫn model không phải là thư mục: {self.model_dir}")
             return

        model_files = [f for f in os.listdir(self.model_dir) if f.endswith(".pth")]
        logger.info(f"🔍 Đang quét model Anti-Spoofing tại: {self.model_dir} (Tìm thấy {len(model_files)} file)")
        
        for model_name in model_files:
            try:
                model_path = os.path.join(self.model_dir, model_name)
                h_input, w_input, model_type, scale = parse_model_name(model_name)
                kernel_size = get_kernel(h_input, w_input)
                
                model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(self.device)
                state_dict = torch.load(model_path, map_location=self.device)
                
                # Xử lý trường hợp model được train bằng DataParallel (có tiền tố 'module.')
                if next(iter(state_dict)).find('module.') >= 0:
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        new_state_dict[k[7:]] = v
                    model.load_state_dict(new_state_dict)
                else:
                    model.load_state_dict(state_dict)
                
                model.eval()
                self.models_cache[model_name] = {
                    'model': model,
                    'h_input': h_input,
                    'w_input': w_input,
                    'scale': scale
                }
                logger.info(f"✅ Đã nạp model Anti-Spoofing: {model_name}")
            except Exception as e:
                logger.error(f"❌ Lỗi khi nạp model {model_name}: {e}")
        
        if not self.models_cache:
            logger.error("❌ KHÔNG NẠP ĐƯỢC MODEL ANTI-SPOOFING NÀO!")
        else:
            logger.info(f"🎉 Hoàn tất nạp {len(self.models_cache)} model Anti-Spoofing.")

    def is_real(self, frame: np.ndarray, bbox_xyxy: np.ndarray) -> tuple[bool, float]:
        """
        Kiểm tra xem khuôn mặt trong bbox có phải là người thật hay không.
        """
        if not TORCH_AVAILABLE:
            return True, 1.0
            
        if not self._models_loaded:
            self.load_models()
            
        if not self.models_cache:
            return True, 1.0
            
        try:
            x1, y1, x2, y2 = bbox_xyxy
            bbox_xywh = [int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))]
            
            prediction = np.zeros((1, 3))
            
            for model_name, m_info in self.models_cache.items():
                param = {
                    "org_img": frame,
                    "bbox": bbox_xywh,
                    "scale": m_info['scale'],
                    "out_w": m_info['w_input'],
                    "out_h": m_info['h_input'],
                    "crop": True if m_info['scale'] is not None else False,
                }
                
                img = self.image_cropper.crop(**param)
                
                # Transform tương tự trong AntiSpoofPredict.predict
                img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float()
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    result = m_info['model'](img_tensor)
                    result = F.softmax(result, dim=1).cpu().numpy()
                    prediction += result
            
            # label 1 là Real Face
            real_score = prediction[0][1] / len(self.models_cache)
            is_real = real_score >= anti_spoof_config.threshold
            
            return bool(is_real), float(real_score)
            
        except Exception as e:
            logger.error(f"Lỗi khi chạy Anti-Spoofing: {e}")
            return True, 1.0

anti_spoof_service = AntiSpoofService.get_instance()
