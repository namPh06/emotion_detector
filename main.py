"""
Real-time emotion recognition using the exported classroom model.

Expected model path:
    ./models/emotion_model_daisee.keras
"""

import os
import sys

import cv2
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.models import load_model

from utils.label_mapping import EMOTION_COLORS, get_ordered_labels


MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "emotion_model_daisee.keras")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "emotion_model_daisee.h5")
LABELS = get_ordered_labels("en")
COLORS = [EMOTION_COLORS[idx] for idx in sorted(EMOTION_COLORS)]


if not os.path.exists(MODEL_PATH):
    print("[ERROR] Model not found.")
    print(f"  -> Put emotion_model_daisee.keras or .h5 here: {MODEL_PATH}")
    print("  -> Train a new model with train_best_model.py or read README.md")
    sys.exit(1)


print("[INFO] Loading model...")
model = load_model(MODEL_PATH)
print("[OK] Model ready.")
IMG_SIZE = tuple(int(dim) for dim in model.input_shape[1:3])
print(f"[OK] Model input size: {IMG_SIZE}")

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
print("[OK] Face detector ready.")


def preprocess_face(frame, x1, y1, x2, y2, pad=20):
    h, w = frame.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None, None, None, None, None
    face = cv2.resize(face, IMG_SIZE)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float32")
    return np.expand_dims(face, 0), x1, y1, x2, y2


def draw_ui(frame, x1, y1, x2, y2, label, conf, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 12, y1), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 6, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )
    bar_w = int((x2 - x1) * conf)
    cv2.rectangle(frame, (x1, y2 + 4), (x2, y2 + 14), (60, 60, 60), -1)
    cv2.rectangle(frame, (x1, y2 + 4), (x1 + bar_w, y2 + 14), color, -1)


def draw_legend(frame):
    for i, (lbl, clr) in enumerate(zip(LABELS, COLORS)):
        y = 30 + i * 28
        cv2.rectangle(frame, (10, y - 14), (26, y + 2), clr, -1)
        cv2.putText(frame, lbl, (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("[INFO] Webcam inference started. Press Q to quit.")

SMOOTH = 5
history = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (200, 145), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    draw_legend(frame)

    for (fx, fy, fw, fh) in faces:
        out = preprocess_face(frame, fx, fy, fx + fw, fy + fh)
        if out[0] is None:
            continue
        face_arr, x1, y1, x2, y2 = out

        preds = model.predict(face_arr, verbose=0)[0]
        history.append(preds)
        if len(history) > SMOOTH:
            history.pop(0)
        avg = np.mean(history, axis=0)
        idx = int(np.argmax(avg))
        conf = float(avg[idx])

        draw_ui(frame, x1, y1, x2, y2, LABELS[idx], conf, COLORS[idx])

    cv2.putText(
        frame,
        "Emotion Recognition | Press Q to quit",
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (120, 120, 120),
        1,
    )
    cv2.imshow("Classroom Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Exited.")
