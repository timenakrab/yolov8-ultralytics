import torch
import cv2
import os
import time
from ultralytics import YOLO

# กำหนดอุปกรณ์ (device) ที่จะใช้
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# โหลดโมเดล
model = YOLO('yolov8n.pt')
model.to(device)

# โฟลเดอร์สำหรับ input และ output
input_video_path = "./videos/ForBiggerFun.mp4"
output_video_path = "./outputs/result_ForBiggerFun.mp4"

# เปิดไฟล์วิดีโอ
cap = cv2.VideoCapture(input_video_path)

# ตรวจสอบว่าวิดีโอเปิดได้หรือไม่
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# อ่านข้อมูลวิดีโอ
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
bitrate = cap.get(cv2.CAP_PROP_BITRATE)  # ดึงค่า bitrate จากวิดีโอต้นทาง
codec = int(cap.get(cv2.CAP_PROP_FOURCC))
codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
print(f'Input video codec: {codec_str}')

# ตรวจสอบว่า bitrate ได้รับการดึงค่าอย่างถูกต้องหรือไม่
if bitrate == -1:
    bitrate = 2000000  # กำหนดค่า default ถ้าไม่สามารถดึงค่าได้

# กำหนดการตั้งค่า output วิดีโอ
fourcc = cv2.VideoWriter_fourcc(*codec_str)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # สามารถเปลี่ยนเป็น 'XVID', 'H264', หรือ 'X264'
# หรือ
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # ใช้ XVID codec
# หรือ
# fourcc = cv2.VideoWriter_fourcc(*'H264')  # ใช้ H.264 codec
# หรือ
# fourcc = cv2.VideoWriter_fourcc(*'X264')  # ใช้ X264 codec

out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), True)

# เริ่มจับเวลา
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # ประมวลผลเฟรมด้วย YOLO
    results = model(frame)
    
    # แสดงผลลัพธ์บนเฟรม
    annotated_frame = results[0].plot()  # ใช้ plot() เพื่อใส่ bounding box บนเฟรม
    
    # บันทึกเฟรมที่ประมวลผลแล้วลงวิดีโอ output
    out.write(annotated_frame)

# ปิดไฟล์วิดีโอ
cap.release()
out.release()

# จับเวลาสิ้นสุด
end_time = time.time()
total_time = end_time - start_time
print(f"Processing complete. Total time: {total_time:.2f} seconds.")
