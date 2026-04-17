import cv2

def list_cameras():
    print("Checking for cameras (indices 0-4)...")
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Index {i}: Found!")
            cap.release()
        else:
            # Try without DSHOW
            cap = cv2.VideoCapture(i, cv2.CAP_ANY)
            if cap.isOpened():
                print(f"Index {i}: Found (Any)!")
                cap.release()
            else:
                print(f"Index {i}: Not found.")

if __name__ == "__main__":
    list_cameras()
