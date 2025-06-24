import os
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "piece_classifier.h5"
TEST_IMAGE_DIR = "dataset/manual"

IMG_SIZE = (64, 128)
CLASSES = ["I", "O", "T", "S", "Z", "L", "J"]

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return img.reshape(1, IMG_SIZE[1], IMG_SIZE[0], 1)

print("\n📊 Resultados por classe:")
for cls in CLASSES:
    folder = os.path.join(TEST_IMAGE_DIR, cls)
    if not os.path.isdir(folder):
        continue

    images = [f for f in os.listdir(folder) if f.endswith((".png", ".jpg"))]
    if not images:
        continue

    test_path = os.path.join(folder, images[0])
    input_img = preprocess_image(test_path)
    pred = model.predict(input_img)[0]
    predicted_idx = np.argmax(pred)
    predicted_class = CLASSES[predicted_idx]

    print(f"- Verdadeiro: {cls:<2} | Predito: {predicted_class} | Confiança: {pred[predicted_idx]:.2f}")