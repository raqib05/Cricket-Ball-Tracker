from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="config.yaml",
    imgsz=640,
    epochs=150,
    batch=8,
    device="mps",
    cache=True
)
