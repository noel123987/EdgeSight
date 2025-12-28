# **EdgeSight – Real-Time Object Detection & Tracking**

EdgeSight is a real-time object detection and tracking system built using **YOLOv8 / YOLOv10** and **OpenCV**. It processes live webcam or video streams and identifies multiple objects with high accuracy while maintaining real-time performance. The project focuses on **speed vs accuracy optimization**, **edge deployability**, and **practical real-world usability**.

---

## 🚀 **Features**

* ✔️ Real-time object detection using YOLOv8 / YOLOv10
* ✔️ Supports Webcam, CCTV feeds, and video files
* ✔️ Multi-object tracking with bounding boxes and class labels
* ✔️ FPS counter to measure real-time performance
* ✔️ Adjustable confidence thresholds
* ✔️ Model optimization ready (ONNX / INT8 / TensorRT)
* ✔️ Edge-device friendly architecture

---

## 🧠 **Use Cases**

* Retail people counting & analytics
* Traffic monitoring and vehicle detection
* Workplace safety gear detection
* Smart surveillance systems
* General computer vision learning & experimentation

---

## 🛠️ **Tech Stack**

* Python
* OpenCV
* Ultralytics YOLOv8 / YOLOv10
* (Optional) Gradio / Flask for UI dashboards

---

## ⚙️ **Installation**

### 1️⃣ Create & activate virtual environment (Recommended)

```
python -m venv venv
source venv/Scripts/activate     # Git Bash
# or
venv\Scripts\activate            # CMD / PowerShell
```

### 2️⃣ Install dependencies

```
pip install ultralytics opencv-python
```

---

## ▶️ **How to Run**

Place your script in `app.py` and run:

```
python app.py
```

The system will:

* Start your webcam (or load video)
* Run YOLO detection
* Show real-time bounding boxes and FPS

---

## 📂 **Project Structure**

```
EdgeSight/
 ├── app.py
 ├── README.md
 ├── requirements.txt
 └── assets/ (optional screenshots/videos)
```

---

## ⚡ **Performance Optimization (Future Enhancements)**

* Convert model to ONNX
* INT8 Quantization
* TensorRT acceleration
* Lighter model variants (YOLO-Nano, YOLO-N)

---

## 📌 **Future Scope**

* Web dashboard using Flask / Gradio
* Object counting and zone-based alerts
* Database logging for analytics
* Deployment to Raspberry Pi / Jetson Nano

---

## 🏆 **Why This Project Matters**

EdgeSight helps understand **real-world AI constraints**, including:

* Accuracy vs Speed trade-offs
* Latency & hardware limitations
* Model optimization and deployment readiness

This makes it highly relevant for **industry projects, interviews, and real deployments**.

---

## 🤝 **Contributions**

Pull requests are welcome. Feel free to open issues for discussion or enhancements.

---

## 📜 **License**

Open-source for educational and project purposes.




