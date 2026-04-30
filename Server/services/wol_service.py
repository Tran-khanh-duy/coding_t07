"""
services/wol_service.py
Wake-on-LAN Service — Đánh thức Mini PC từ xa qua gói tin Magic Packet.

Cấu hình MAC address của từng Mini PC trong config.py (danh sách MINI_PCS).
Khi Server khởi động, tự động gửi Magic Packet đến tất cả thiết bị trong danh sách.
"""
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from loguru import logger


# ─── Magic Packet ─────────────────────────────────────────────────────────────

def send_magic_packet(mac_address: str, broadcast: str = "255.255.255.255", port: int = 9) -> bool:
    """
    Gửi Magic Packet (WOL) tới địa chỉ MAC được chỉ định.

    Args:
        mac_address: Địa chỉ MAC dạng "AA:BB:CC:DD:EE:FF" hoặc "AA-BB-CC-DD-EE-FF"
        broadcast:   Địa chỉ broadcast LAN (mặc định 255.255.255.255)
        port:        Port WOL, thường là 9 hoặc 7

    Returns:
        True nếu gói tin được gửi thành công, False nếu lỗi.
    """
    try:
        # Chuẩn hóa MAC — loại bỏ dấu phân cách
        clean_mac = mac_address.upper().replace(":", "").replace("-", "").replace(".", "")
        if len(clean_mac) != 12:
            raise ValueError(f"MAC address không hợp lệ: '{mac_address}' (cần 12 ký tự hex)")

        # Tạo Magic Packet: 6 byte 0xFF + 16 lần lặp lại MAC (102 bytes tổng)
        mac_bytes = bytes.fromhex(clean_mac)
        magic_packet = b"\xff" * 6 + mac_bytes * 16

        # Gửi qua UDP Broadcast
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(magic_packet, (broadcast, port))

        logger.success(f"🔌 WOL Magic Packet đã gửi → {mac_address} (broadcast: {broadcast}:{port})")
        return True

    except Exception as e:
        logger.error(f"❌ Gửi WOL thất bại [{mac_address}]: {e}")
        return False


# ─── Mini PC Registry ─────────────────────────────────────────────────────────

@dataclass
class MiniPCDevice:
    """Thông tin một thiết bị Mini PC."""
    name:         str                    # Ví dụ: "Mini PC KTX E4 - Tầng 4"
    mac_address:  str                    # Ví dụ: "AA:BB:CC:DD:EE:01"
    ip_address:   str = ""               # IP tĩnh (nếu có), để kiểm tra online
    location:     str = ""               # Ví dụ: "KTX E4 - Tầng 4"
    broadcast:    str = "255.255.255.255" # Broadcast tùy theo subnet
    wol_port:     int = 9
    is_online:    bool = False
    last_seen:    float = 0.0
    device_name:  str = ""               # device_name trùng với edge_config.device_name


@dataclass
class WOLConfig:
    """Cấu hình danh sách Mini PC cho WOL."""
    devices: List[MiniPCDevice] = field(default_factory=list)
    auto_wake_on_startup: bool = True    # Tự động đánh thức khi Server khởi động
    wake_delay_sec: float = 0.5          # Delay giữa các gói tin
    ping_timeout_sec: float = 1.0
    online_timeout_sec: float = 30.0     # Sau N giây không thấy = offline


# ─── WOL Service ──────────────────────────────────────────────────────────────

class WOLService:
    """
    Dịch vụ Wake-on-LAN trung tâm.

    Sử dụng:
        wol_service.wake_all()           # Đánh thức tất cả Mini PC
        wol_service.wake(name="...")      # Đánh thức 1 thiết bị cụ thể
        wol_service.update_online_status(device_name, ip) # Gọi khi nhận edge_status
    """

    def __init__(self, config: WOLConfig = None):
        self._config = config or WOLConfig()
        self._lock = threading.Lock()
        self._status_callbacks: List[Callable] = []  # UI callbacks khi trạng thái thay đổi

    # ─── Cấu hình ─────────────────────────────

    def set_devices(self, devices: List[MiniPCDevice]):
        """Cập nhật danh sách thiết bị."""
        with self._lock:
            self._config.devices = devices

    @property
    def devices(self) -> List[MiniPCDevice]:
        with self._lock:
            return list(self._config.devices)

    def add_device(self, device: MiniPCDevice):
        with self._lock:
            # Tránh trùng lặp
            existing = [d for d in self._config.devices if d.mac_address == device.mac_address]
            if not existing:
                self._config.devices.append(device)

    # ─── Wake ──────────────────────────────────

    def wake_all(self, async_mode: bool = True):
        """Gửi Magic Packet tới TẤT CẢ thiết bị đã đăng ký."""
        if not self._config.devices:
            logger.warning("⚠️ WOL: Chưa cấu hình thiết bị nào.")
            return

        def _do_wake_all():
            logger.info(f"⚡ WOL: Đang đánh thức {len(self._config.devices)} Mini PC...")
            for device in self._config.devices:
                if not device.is_online:  # Chỉ wake nếu đang offline
                    self._send_wol(device)
                    time.sleep(self._config.wake_delay_sec)
                else:
                    logger.debug(f"✅ WOL: {device.name} đã online, bỏ qua.")

        if async_mode:
            threading.Thread(target=_do_wake_all, name="WOL-WakeAll", daemon=True).start()
        else:
            _do_wake_all()

    def wake(self, name: str = None, mac: str = None, async_mode: bool = True) -> bool:
        """
        Đánh thức một Mini PC cụ thể theo tên hoặc MAC address.

        Returns:
            True nếu tìm thấy và gửi packet.
        """
        with self._lock:
            if mac:
                target = next((d for d in self._config.devices
                               if d.mac_address.upper().replace(":", "") == mac.upper().replace(":", "")), None)
            elif name:
                target = next((d for d in self._config.devices
                               if name.lower() in d.name.lower() or name.lower() in d.location.lower()), None)
            else:
                return False

        if not target:
            logger.warning(f"⚠️ WOL: Không tìm thấy thiết bị '{name or mac}'")
            return False

        if async_mode:
            threading.Thread(
                target=self._send_wol,
                args=(target,),
                name=f"WOL-{target.name}",
                daemon=True
            ).start()
        else:
            self._send_wol(target)
        return True

    def _send_wol(self, device: MiniPCDevice):
        """Gửi Magic Packet tới một thiết bị."""
        ok = send_magic_packet(
            mac_address=device.mac_address,
            broadcast=device.broadcast,
            port=device.wol_port,
        )
        if ok:
            logger.info(f"  ↳ {device.name} | MAC: {device.mac_address}")

    # ─── Online Status Tracking ─────────────────

    def update_online_status(self, device_name: str, ip_address: str = ""):
        """
        Gọi khi nhận được tín hiệu edge_status từ Mini PC.
        Đánh dấu thiết bị đó là ONLINE.
        """
        with self._lock:
            now = time.time()
            for device in self._config.devices:
                if (device.device_name and device.device_name == device_name) or \
                   (ip_address and device.ip_address == ip_address):
                    was_offline = not device.is_online
                    device.is_online = True
                    device.last_seen = now
                    if ip_address:
                        device.ip_address = ip_address

                    if was_offline:
                        logger.success(f"🟢 Mini PC ONLINE: {device.name} ({ip_address})")
                        self._notify_callbacks()
                    return

    def check_timeouts(self):
        """
        Kiểm tra thiết bị nào đã lâu không gửi heartbeat → đánh dấu OFFLINE.
        Gọi từ một timer nền.
        """
        now = time.time()
        changed = False
        with self._lock:
            for device in self._config.devices:
                if device.is_online:
                    if now - device.last_seen > self._config.online_timeout_sec:
                        device.is_online = False
                        logger.warning(f"🔴 Mini PC OFFLINE: {device.name}")
                        changed = True
        if changed:
            self._notify_callbacks()

    def get_status_summary(self) -> List[dict]:
        """Trả về danh sách trạng thái để hiển thị trên UI."""
        with self._lock:
            return [
                {
                    "name":        d.name,
                    "mac":         d.mac_address,
                    "ip":          d.ip_address,
                    "location":    d.location,
                    "is_online":   d.is_online,
                    "last_seen":   d.last_seen,
                }
                for d in self._config.devices
            ]

    # ─── UI Callbacks ──────────────────────────

    def add_status_callback(self, callback: Callable):
        """Đăng ký callback được gọi khi trạng thái online/offline thay đổi."""
        self._status_callbacks.append(callback)

    def _notify_callbacks(self):
        for cb in self._status_callbacks:
            try:
                cb()
            except Exception as e:
                logger.debug(f"WOL callback error: {e}")


# ─── Singleton ────────────────────────────────────────────────────────────────

wol_service = WOLService()
