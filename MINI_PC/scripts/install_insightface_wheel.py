import sys
import os
import subprocess
import urllib.request
import platform

def install_insightface_wheel():
    print("\n" + "="*50)
    print("   INSIGHTFACE REPAIR TOOL (PRE-BUILT WHEEL)")
    print("="*50)

    # 1. Check if Windows
    if platform.system() != "Windows":
        print("Error: This tool is only for Windows.")
        return False

    # 2. Get Python Version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"[*] Detected Python: {py_version}")

    wheels = {
        "3.10": "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp310-cp310-win_amd64.whl",
        "3.11": "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp311-cp311-win_amd64.whl",
        "3.12": "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp312-cp312-win_amd64.whl"
    }

    if py_version not in wheels:
        print(f"Error: Pre-built wheel not found for Python {py_version}.")
        print("Supported versions: 3.10, 3.11, 3.12")
        return False

    url = wheels[py_version]
    filename = url.split("/")[-1]
    
    # 3. Download
    print(f"[*] Downloading: {filename}")
    print(f"[*] From: {url}")
    try:
        urllib.request.urlretrieve(url, filename)
        print("[OK] Download complete.")
    except Exception as e:
        print(f"Error downloading wheel: {e}")
        return False

    # 4. Install
    print(f"[*] Installing {filename}...")
    try:
        # We use sys.executable to ensure we install into the current venv
        subprocess.check_call([sys.executable, "-m", "pip", "install", filename])
        print(f"[OK] {filename} installed successfully!")
        
        # Cleanup
        os.remove(filename)
        return True
    except Exception as e:
        print(f"Error installing wheel: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return False

if __name__ == "__main__":
    success = install_insightface_wheel()
    if not success:
        print("\n" + "!"*50)
        print("  FAILED TO INSTALL INSIGHTFACE WHEEL.")
        print("  Please try to install Microsoft C++ Build Tools:")
        print("  https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        print("!"*50)
        sys.exit(1)
    else:
        sys.exit(0)
