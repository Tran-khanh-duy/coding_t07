import psutil
import requests
import time
import threading
import logging
import os

logger = logging.getLogger("HardwareMonitor")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

def get_cpu_temp():
    """Hỗ trợ lấy nhiệt độ (Tùy thuộc HĐH và phần cứng)"""
    try:
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return temps['coretemp'][0].current
            elif temps:
                # Trả về cảm biến đầu tiên tìm được
                return list(temps.values())[0][0].current
    except Exception:
        pass
    return 45.0  # Mặc định / giả định cho môi trường không đọc được (như Windows không chạy quyền Admin)

def hardware_monitor_loop(server_url: str, edge_id: str):
    """Vòng lặp thu thập và POST lên Server mỗi 2 giây"""
    logger.info(f"Khởi động Hardware Monitor. Gửi tới: {server_url} | Edge ID: {edge_id}")
    while True:
        try:
            mem = psutil.virtual_memory()
            payload = {
                "edge_id": edge_id,
                "cpu_percent": psutil.cpu_percent(interval=1),
                "ram_percent": mem.percent,
                "ram_used_gb": round(mem.used / (1024 ** 3), 1),
                "ram_total_gb": round(mem.total / (1024 ** 3), 1),
                "temperature": get_cpu_temp()
            }
            # Gửi lên API FastAPI
            print(f"[EDGE DEBUG] Đang POST JSON: {payload} lên {server_url}/api/hardware/metrics")
            resp = requests.post(f"{server_url}/api/hardware/metrics", json=payload, timeout=2)
            print(f"[EDGE DEBUG] HTTP Status Code Server trả về: {resp.status_code}")
            logger.debug(f"Đã gửi telemetry: CPU {payload['cpu_percent']}% | RAM {payload['ram_percent']}% | Temp {payload['temperature']}°C")
        except Exception as e:
            logger.error(f"Gửi telemetry thất bại (Sẽ thử lại sau 1 phút): {e}")
        
        # BẮT BUỘC: Sleep 60 giây theo yêu cầu để giới hạn tần suất gửi 1 phút/lần
        time.sleep(60)

if __name__ == "__main__":
    # Lấy thông số từ biến môi trường hoặc chạy mặc định
    SERVER_URL = os.getenv("EDGE_SERVER_URL", "http://127.0.0.1:9696")
    EDGE_ID = os.getenv("EDGE_CAMERA_ID", "MINI_PC_01")
    hardware_monitor_loop(SERVER_URL, EDGE_ID)
