import requests

# Khai báo thông tin Bot (Sẽ chuyển sang file .env ở Bước 4 để bảo mật)
BOT_TOKEN = "8210386728:AAFCh-nmQOul9g3tIrzMcNIypNf4x6M8XdE"
CHAT_ID = "6000467659"

def send_telegram_msg(message):
    """Gửi tin nhắn qua Telegram Bot"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }
    try:
        # Gửi request với timeout để không làm treo hệ thống nếu rớt mạng
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            print("Đã gửi thông báo Telegram thành công.")
        else:
            print(f"Lỗi gửi tin: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Lỗi kết nối mạng khi gửi Telegram: {e}")