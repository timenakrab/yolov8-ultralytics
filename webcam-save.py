import warnings

# ละเว้นคำเตือนเกี่ยวกับ AVFoundation
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description="YOLO Object Detection")
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],help='Device to use for inference (default: cpu)')
args = parser.parse_args()

device = args.device
if device == 'cuda' and not torch.cuda.is_available():
    print("CUDA is not available, falling back to CPU")
    device = 'cpu'
elif device == 'mps' and not torch.backends.mps.is_available():
    print("MPS is not available, falling back to CPU")
    device = 'cpu'

# กำหนดอุปกรณ์ (device) ที่จะใช้
# device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# โหลดโมเดล
model = YOLO('yolov8n.pt')
model.to(device)

# สร้างโฟลเดอร์สำหรับบันทึกภาพ
output_directory = "./webcam/"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# เปิดกล้องเว็บแคม
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # ประมวลผลเฟรมด้วย YOLO
    results = model(frame)
    
    # แสดงผลลัพธ์บนเฟรม
    annotated_frame = results[0].plot()  # ใช้ plot() เพื่อใส่ bounding box บนเฟรม
    
    # แสดงภาพแบบเรียลไทม์
    cv2.imshow('Webcam', annotated_frame)
    
    # บันทึกเฟรมที่ประมวลผลแล้วลงในโฟลเดอร์ "webcam"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
    output_filepath = os.path.join(output_directory, f"{timestamp}.jpg")
    cv2.imwrite(output_filepath, annotated_frame)
    
    # กด 'q' เพื่อออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่างแสดงผล
cap.release()
cv2.destroyAllWindows()
