import os
import sys
import traceback
from pathlib import Path

# Add current dir to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

print("--- DIAGNOSTIC START ---")

# 1. Check Config
try:
    from config import anti_spoof_config
    print(f"✅ Config Load: SUCCESS (enabled={anti_spoof_config.enabled})")
except Exception:
    print("❌ Config Load: FAILED")
    traceback.print_exc()

# 2. Check Folder Structure
as_root = ROOT / "Silent-Face-Anti-Spoofing-master"
print(f"Check Folder: {as_root}")
if as_root.exists():
    print("✅ Folder Silent-Face-Anti-Spoofing-master: EXISTS")
    resources = as_root / "resources" / "anti_spoof_models"
    if resources.exists():
        files = list(resources.glob("*.pth"))
        print(f"✅ Models Folder: EXISTS ({len(files)} models found)")
    else:
        print("❌ Models Folder: MISSING")
else:
    print("❌ Folder Silent-Face-Anti-Spoofing-master: MISSING")

# 3. Check Torch
try:
    import torch
    print(f"✅ Torch Version: {torch.__version__}")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
except Exception:
    print("❌ Torch Load: FAILED")
    traceback.print_exc()

# 4. Check Service Import (Internal)
try:
    print("Attempting to import anti_spoof_service...")
    from services.anti_spoof_service import anti_spoof_service
    print(f"✅ Service Import: SUCCESS")
    print(f"✅ Loaded Models: {list(anti_spoof_service.models_cache.keys())}")
except Exception:
    print("❌ Service Import: FAILED")
    traceback.print_exc()

# 5. Check FaceEngine Import (External)
try:
    print("Attempting to import face_engine and check ANTI_SPOOF_AVAILABLE...")
    from services.face_engine import ANTI_SPOOF_AVAILABLE
    print(f"✅ FaceEngine ANTI_SPOOF_AVAILABLE: {ANTI_SPOOF_AVAILABLE}")
except Exception:
    print("❌ FaceEngine Import: FAILED")
    traceback.print_exc()

print("--- DIAGNOSTIC END ---")
