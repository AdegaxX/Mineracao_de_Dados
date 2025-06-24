import cv2
import numpy as np
import tensorflow as tf
from config import PIECE_REGION
from mss import mss

CLASSES = ["I", "O", "T", "S", "Z", "L", "J"]
MODEL_PATH = "piece_classifier.keras"  # se você salvou assim

model = tf.keras.models.load_model(MODEL_PATH)

def detect_piece(show_window=False):
    with mss() as sct:
        img = sct.grab(PIECE_REGION)
        frame = np.array(img)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    resized = cv2.resize(gray, (128, 64))

    # Normalização e reshape para (1, 64, 128, 1)
    normalized = resized.astype("float32") / 255.0
    input_img = np.expand_dims(normalized, axis=(0, -1))

    predictions = model.predict(input_img, verbose=0)[0]
    class_idx = np.argmax(predictions)
    confidence = predictions[class_idx]
    piece_type = CLASSES[class_idx]

    if show_window:
        cv2.imshow("Detecção da Peça", cv2.resize(resized, (128, 128)))
        cv2.waitKey(1)

    return piece_type, class_idx, confidence
