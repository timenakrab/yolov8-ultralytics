import torch
import os
from ultralytics import YOLO

# กำหนดอุปกรณ์ (device) ที่จะใช้
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO('yolov8n.pt')
model.to(device)

input_directory = "./images/"
output_directory = "./outputs/"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in os.listdir(input_directory):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        filepath = os.path.join(input_directory, filename)
        results = model(filepath)

        output_filepath = os.path.join(output_directory, f"result_{filename}")

        for result in results:
            result.save(output_filepath)

print("Processing complete.")
