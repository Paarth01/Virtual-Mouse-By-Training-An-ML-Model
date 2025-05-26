import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Parameters
DATASET_DIR = 'dataset'
EPOCHS = 30
BATCH_SIZE = 32
MODEL_PATH = 'gesture_model.h5'

# 1. Load all CSVs and assign labels
gesture_csvs = glob(os.path.join(DATASET_DIR, '*', '*_landmarks.csv'))
gesture_names = [os.path.basename(os.path.dirname(csv)) for csv in gesture_csvs]
gesture_to_label = {name: idx for idx, name in enumerate(sorted(gesture_names))}

X = []
y = []
for csv_path in gesture_csvs:
    gesture = os.path.basename(os.path.dirname(csv_path))
    label = gesture_to_label[gesture]
    df = pd.read_csv(csv_path)
    X.append(df.values)
    y += [label] * len(df)

X = np.vstack(X)
y = np.array(y)

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. One-hot encode labels
num_classes = len(gesture_to_label)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# 4. Build simple MLP model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train
model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=EPOCHS, batch_size=BATCH_SIZE)

# 6. Save model
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
print("Gesture label mapping:")
for gesture, idx in gesture_to_label.items():
    print(f"{idx}: {gesture}")