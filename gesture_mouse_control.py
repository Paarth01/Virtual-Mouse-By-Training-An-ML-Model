import cv2
import numpy as np
import tensorflow as tf
import pyautogui
import os
import mediapipe as mp
from collections import deque, Counter

# Parameters
model_path = 'gesture_model.h5'
gesture_labels = sorted(os.listdir('dataset'))  # Assumes folders are gesture names
SMOOTHING_WINDOW = 15  # Number of frames to smooth over
pred_history = deque(maxlen=SMOOTHING_WINDOW)

# Load model
model = tf.keras.models.load_model(model_path)

# Mouse mapping (customized as per your request)
gesture_to_action = {
    'fist': lambda: pyautogui.rightClick(),
    'pinch': lambda: pyautogui.leftClick(),
    # 'index_finger' will be handled separately for continuous mouse movement
}

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Open webcam with error handling
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)
x1, y1, x2, y2 = 100, 100, 400, 400

print("Press 'q' to quit.")

screen_w, screen_h = pyautogui.size()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read from webcam.")
        break

    # Draw ROI rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]

    # Mediapipe hand landmark extraction
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    results = hands.process(roi_rgb)

    gesture_pred = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmark_points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
        if landmark_points.shape[0] == 63:
            input_data = np.expand_dims(landmark_points, axis=0)
            pred = model.predict(input_data, verbose=0)
            pred_idx = np.argmax(pred)
            gesture_pred = gesture_labels[pred_idx]
            pred_history.append(gesture_pred)
            # Smoothing: majority vote
            most_common_gesture, count = Counter(pred_history).most_common(1)[0]
            if count > SMOOTHING_WINDOW // 2:
                if most_common_gesture == 'index_finger':
                    index_tip = hand_landmarks.landmark[8]
                    mouse_x = int(index_tip.x * screen_w)
                    mouse_y = int(index_tip.y * screen_h)
                    pyautogui.moveTo(mouse_x, mouse_y, duration=0.1)
                elif most_common_gesture in gesture_to_action:
                    gesture_to_action[most_common_gesture]()
            cv2.putText(frame, f"Gesture: {most_common_gesture}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        else:
            pred_history.clear()
            cv2.putText(frame, "Hand not fully detected", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    else:
        pred_history.clear()
        cv2.putText(frame, "No hand detected", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()

