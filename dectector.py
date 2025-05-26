import cv2
import os
import mediapipe as mp
import csv
import numpy as np

# Parameters
gesture_name = "index_finger"  # Change this for each gesture (e.g., "fist", "thumbs_up")
save_dir = f"dataset/{gesture_name}"
os.makedirs(save_dir, exist_ok=True)

# CSV file to save landmarks
csv_path = os.path.join(save_dir, f"{gesture_name}_landmarks.csv")
write_header = not os.path.exists(csv_path)

cap = cv2.VideoCapture(0)
img_count = 0
max_images = 200  # Number of images to capture per gesture

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

print(f"Collecting images for gesture: {gesture_name}")
print("Press 'c' to capture and 'q' to quit.")

# Prepare CSV header (21 landmarks * 3 coords)
header = []
for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]

with open(csv_path, mode='a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if write_header:
        writer.writerow(header)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw a rectangle to guide hand placement
        x1, y1, x2, y2 = 100, 100, 400, 400
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        roi = frame[y1:y2, x1:x2]

        # Convert ROI to RGB for Mediapipe
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = hands.process(roi_rgb)

        cv2.imshow("Frame", frame)
        cv2.imshow("ROI", roi)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):
            img_path = os.path.join(save_dir, f"{gesture_name}_{img_count}.jpg")
            cv2.imwrite(img_path, roi)

            # Extract landmarks if present
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmark_row = []
                for lm in hand_landmarks.landmark:
                    landmark_row += [lm.x, lm.y, lm.z]
                writer.writerow(landmark_row)
                print(f"Captured {img_count+1} images with landmarks")

                # >>>> USE landmark_points HERE for live ML prediction or processing <<<<
                landmark_points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                # Example: print or pass to ML model
                # prediction = my_model.predict([landmark_points])
                # print('Prediction:', prediction)
            else:
                print(f"No hand detected for image {img_count+1}, not saving landmarks.")

            img_count += 1
            if img_count >= max_images:
                print("Image collection complete for this gesture.")
                break
        elif key & 0xFF == ord('q'):
            break

cap.release()
hands.close()
cv2.destroyAllWindows()