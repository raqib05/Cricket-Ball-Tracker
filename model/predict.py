from ultralytics import YOLO

model = YOLO("../runs/detect/train3/weights/best.pt")

results = model(source="../cricket_short.mp4", show=True, conf=0.25, save=True)