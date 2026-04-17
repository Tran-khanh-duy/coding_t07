import cv2

def check_cameras():
    print("🔍 Đang quét danh sách camera khả dụng trên máy...")
    available_indices = []
    # Quét từ 0 đến 10
    for i in range(11):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"✅ Camera INDEX [{i}]: ĐANG HOẠT ĐỘNG")
                available_indices.append(str(i))
            cap.release()
        else:
            # Thử backend mặc định
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"✅ Camera INDEX [{i}]: ĐANG HOẠT ĐỘNG (MSMF)")
                available_indices.append(str(i))
                cap.release()
    
    if not available_indices:
        print("❌ KHÔNG tìm thấy camera nào đang kết nối!")
    else:
        print("-" * 30)
        print(f"💡 Gợi ý cấu hình .env.edge:")
        print(f"EDGE_CAMERA_SOURCE={','.join(available_indices)}")
        print("-" * 30)

if __name__ == "__main__":
    check_cameras()
    print("\n[XONG] Hay copy danh sach Index phia tren vao file .env.edge")
    input("Nhan Enter de thoat...")
