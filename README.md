# 🖱️ Virtual Mouse Using a Trained ML Model

A virtual mouse system built using Python, OpenCV, and MediaPipe. This application enables users to control mouse movements and actions through hand gestures captured via a webcam.

---

## 📌 Features

- 👆 **Cursor Movement**: Track your index finger to move the mouse cursor.
- 🤏 **Left Click**: Pinch your index finger and thumb.
- ✊ **Right Click**: Make a closed fist.
- 🧠 **Custom Gesture Recognition**: Train your own gestures with a custom ML model.

---

## 🧰 Tech Stack

| Technology     | Purpose                                |
|----------------|----------------------------------------|
| Python         | Core programming language              |
| OpenCV         | Real-time video processing             |
| MediaPipe      | Hand landmarks and tracking            |
| Scikit-learn   | Machine learning model training        |
| PyAutoGUI      | Mouse control automation               |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Paarth01/Virtual-Mouse-By-Training-An-ML-Model.git
cd Virtual-Mouse-By-Training-An-ML-Model
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install opencv-python
```

### 3. Run the Application

```bash
python main.py
```

## 📁 Project Structure

```bash
.
├── dataset
    ├── fist
    ├── index_finger
    ├── pinch
├── detector.py                  # Script to collect gesture data
├── train_gesture_model.py       # Train and export ML model
├── gesture_mouse_control.py     # Virtual mouse application
├── gesture_model.h5             # Trained ML model
├── requirements.txt             # Python dependencies
└── README.md
```
