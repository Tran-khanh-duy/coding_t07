import os
import sys
from pathlib import Path

# Thêm thư mục gốc vào path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
env_file = ROOT / ".env.edge"
if env_file.exists():
    load_dotenv(env_file, override=True)
    print(f"Loaded {env_file}")

from config import edge_config

print("--- Camera Configuration Test ---")
print(f"Device Name: {edge_config.device_name}")
print(f"Camera IDs: {edge_config.camera_id}")
print(f"Camera Sources: {edge_config.camera_source}")
print(f"Process Every N: {edge_config.process_every_n}")
print("\nParsed Camera List:")
for cam in edge_config.camera_list:
    print(f" - ID: {cam['id']}, Source: {cam['source']}")

assert len(edge_config.camera_list) == 5, f"Expected 5 cameras, got {len(edge_config.camera_list)}"
print("\n✅ Verification SUCCESS: 5 cameras detected.")
