import requests
import os
import sys

def download_models():
    models = [
        "1.0_120x120_MiniFASNetV2.pth",
        "2.7_80x80_MiniFASNetV2.pth",
        "4_80x80_MiniFASNetV1SE.pth"
    ]
    base_url = "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/"
    # Get the project root assuming script is in MINI_PC/scripts/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(project_root, "Silent-Face-Anti-Spoofing-master", "resources", "anti_spoof_models")

    if not os.path.exists(save_dir):
        print(f"Creating directory: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)

    print("--- Downloading Anti-Spoofing Models via Python ---")
    
    for m in models:
        url = base_url + m
        path = os.path.join(save_dir, m)
        if os.path.exists(path):
            print(f"[SKIP] {m} already exists.")
            continue
            
        print(f"[SYNC] Downloading {m}...")
        try:
            r = requests.get(url, timeout=120, stream=True)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[DONE] Successfully downloaded {m}.")
        except Exception as e:
            print(f"[ERROR] Failed to download {m}: {e}")
            print(f"Please download manually from: {url}")

if __name__ == "__main__":
    download_models()
