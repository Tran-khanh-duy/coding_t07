import sys
import os
from pathlib import Path

# Add current dir to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import anti_spoof_config
print(f"DEBUG: anti_spoof_config.model_dir = {anti_spoof_config.model_dir}")
print(f"DEBUG: Path exists? {anti_spoof_config.model_dir.exists()}")
if anti_spoof_config.model_dir.exists():
    print(f"DEBUG: Files in dir: {os.listdir(anti_spoof_config.model_dir)}")

try:
    from services.anti_spoof_service import anti_spoof_service
    print(f"✅ AntiSpoofService initialization: SUCCESS")
    print(f"✅ Models loaded: {list(anti_spoof_service.models_cache.keys())}")
except Exception as e:
    print(f"❌ AntiSpoofService initialization: FAILED")
    import traceback
    traceback.print_exc()
