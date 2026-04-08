import os
import sys
import cv2
import numpy as np
from loguru import logger
from pathlib import Path

# Thêm path để import config
sys.path.insert(0, str(Path(__file__).parent.parent))
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
ANTI_SPOOF_ROOT = Path(__file__).parent.parent / "Silent-Face-Anti-Spoofing-master"
if str(ANTI_SPOOF_ROOT) not in sys.path:
    sys.path.insert(0, str(ANTI_SPOOF_ROOT))

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
        self.model_dir = str(anti_spoof_config.model_dir)
        
        # Khởi tạo predictor và cropper
        self.predictor = AntiSpoofPredict(anti_spoof_config.device_id if anti_spoof_config.device_id >= 0 else 0)
        self.image_cropper = CropImage()
        
        # Cache các model để tránh load lại
        self.models_cache = {}
        self._load_all_models()

    def _load_all_models(self):
        logger.info(f"🔍 Đang kiểm tra thư mục model: {self.model_dir}")
        if not TORCH_AVAILABLE:
            logger.warning("❌ Torch không khả dụng, không thể load model Anti-Spoofing.")
            return
            
        if not os.path.exists(self.model_dir):
            logger.error(f"❌ Thư mục model KHÔNG TỒN TẠI: {self.model_dir}")
            return

        model_files = [f for f in os.listdir(self.model_dir) if f.endswith(".pth")]
        logger.info(f"🔍 Tìm thấy {len(model_files)} file .pth trong {self.model_dir}")
        
        if not model_files:
            logger.error(f"❌ KHÔNG tìm thấy tệp mô hình (.pth) trong {self.model_dir}")
            logger.warning("Vui lòng chạy script 'setup_anti_spoof_models.ps1' để tải mô hình.")
            return

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
                logger.info(f"✅ Đã nạp thành công model: {model_name}")
            except Exception as e:
                logger.error(f"❌ Lỗi khi nạp model {model_name}: {e}")

        if self.models_cache:
            logger.success(f"🚀 Anti-Spoofing Service: ĐÃ SẴN SÀNG ({len(self.models_cache)} models)")
        else:
            logger.warning("⚠️ Anti-Spoofing Service: Không có model nào được nạp.")

    def is_real(self, frame: np.ndarray, bbox_xyxy: np.ndarray) -> tuple[bool, float]:
        """
        Kiểm tra xem khuôn mặt trong bbox có phải là người thật hay không.
        """
        if not TORCH_AVAILABLE or not self.models_cache:
            # Fallback: Nếu không có torch hoặc không có model, mặc định là THẬT để không chặn người dùng
            # Tuy nhiên thỉnh thoảng log lại để nhắc nhở
            if not self.models_cache:
                logger.warning("Anti-Spoofing đang tạm tắt do thiếu file models (.pth)")
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
