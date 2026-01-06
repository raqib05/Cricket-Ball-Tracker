# 🏏 Cricket Ball Tracker

A deep learning + computer vision project that detects and tracks a cricket ball frame‑by‑frame in match footage. The goal of this project is to explore how modern object detection models work in practice, understand their limitations, and build an end‑to‑end video inference pipeline.

---

## 🚀 Project Overview

This project uses a CNN‑based object detection model to locate the cricket ball in each video frame and a tracking algorithm to maintain consistent ball identity across frames. It was built as a hands‑on learning project to gain real experience with:

* Training and running deep learning object detection models
* Real‑time video inference
* Multi‑object tracking
* Understanding how data quality and model complexity impact performance

While the tracker works well in many scenarios, it can struggle in complex situations such as occlusions, fast motion, motion blur, or visually cluttered backgrounds — highlighting why high‑quality data and more advanced models are so important.

---

## 🧠 Key Concepts & Tech Stack

* **Deep Learning**: CNN‑based object detection
* **Model**: YOLOv8 (lightweight variant)
* **Tracking**: BoT‑SORT
* **Computer Vision**: OpenCV
* **Frameworks & Tools**:

  * PyTorch
  * Ultralytics YOLO
  * NumPy
  * Python

---

## 🏋️ Training the Model

1. Prepare and label your dataset in YOLO format
2. Update dataset paths and config files
3. Train the model

Training deep learning models is **very time‑consuming**, especially with large datasets or powerful GPUs. This project made that reality very clear 😅.

---

## 🎥 Running Inference & Tracking

The pipeline:

1. Reads video frames
2. Runs object detection on each frame
3. Applies tracking to maintain ball identity
4. Outputs annotated video with tracked ball

---

## ⚠️ Limitations

* Struggles with heavy occlusion (bat, player, crowd)
* Fast motion and motion blur can reduce detection accuracy
* Small object size makes detection challenging
* Performance is highly dependent on data quality

These limitations motivated further exploration into better datasets, longer training, and more advanced learning‑based approaches.

---

## 🔮 Future Work

* Build a **Chrome extension** to enable real‑time ball tracking while watching live or recorded matches
* Improve dataset size and diversity
* Experiment with larger or custom‑trained models

---



