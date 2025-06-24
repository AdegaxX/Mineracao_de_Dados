import cv2
import numpy as np
import tensorflow as tf
import mss

model = tf.keras.models.load_model("piece_classifier.h5")
IMG_SIZE = (64, 128)

PIECE_CLASSES = ["I", "O", "T", "S", "Z", "L", "J"]

def capture_piece_region():
    region = {"top": 155, "left": 880, "width": 275, "height": 510}
    with mss.mss() as sct:
        img = np.array(sct.grab(region))
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img.reshape((1, 64, 128, 3))

def predict_piece():
    img = capture_piece_region()
    pred = model.predict(img)[0]
    idx = np.argmax(pred)
    return PIECE_CLASSES[idx], idx

if __name__ == "__main__":
    piece, class_id = predict_piece()
    print(f"Peça detectada: {piece} (classe {class_id})")
