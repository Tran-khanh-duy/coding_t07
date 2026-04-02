import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(r"c:\Users\QUOC KHAI\Desktop\face_attendance")))

from services.face_engine import face_engine

print("Loading model...")
res = face_engine.load_model()
print("Model loaded:", res)

cap = cv2.VideoCapture(0)
# warm up camera
for i in range(15):
    cap.read()
    
ret, frame = cap.read()
if ret:
    print(f"Frame shape: {frame.shape}")
    cv2.imwrite("test_frame.jpg", frame)
    faces = face_engine.detect_faces(frame)
    print(f"Faces detected: {len(faces)}")
else:
    print("Could not read frame from camera")
cap.release()
