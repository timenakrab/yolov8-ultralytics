import torch
import os
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
