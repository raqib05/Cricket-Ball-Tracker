from ultralytics import YOLO
import cv2

def tracker():
    video_path = "../cricket_short.mp4"
    model_path = "../runs/detect/train3/weights/best.pt"

    model = YOLO(model_path)

    results = model.track(
        source=video_path,
        imgsz=640,
        tracker="botsort.yaml",
        conf=0.15,
        persist=True,
        stream=True,  
        show=False     
    )

    for f in results:
        if f.boxes is None:
            continue

        img = f.orig_img.copy()

        for box in f.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            class_name = model.names[cls]

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{class_name} ID:{track_id} {conf:.2f}"
            cv2.putText(
                img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
            )

        cv2.imshow("YOLOv8 + BoT-SORT", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

tracker()
