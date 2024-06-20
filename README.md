## POC YOLOv8 by ultralytics

`pip install ultralytics torch torchvision torchaudio tensorflow-macos tensorflow-metal opencv-python opencv-python-headless`

`pip install --upgrade cryptography urllib3`

https://chatgpt.com/c/e3deaf49-250c-4d72-b810-680558dc6108
https://chatgpt.com/share/a4d4a073-b5a2-4a0a-b13e-7258f870690b

### change to torch-nightly

conda activate torch-nightly

### change to default

conda activate base

### RUN (cpu, cuda, mps)

python images.py --device cpu
python videos.py --device cpu
python webcam-save.py --device cpu
python webcam-notsave.py --device cpu
