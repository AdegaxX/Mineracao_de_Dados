import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configurações
INPUT_DIR = "dataset/manual"
OUTPUT_DIR = "dataset/expanded"
IMG_SIZE = (64, 128)
CLASSES = ["I", "O", "T", "S", "Z", "L", "J"]
COPIAS_EXTRA = 70  # Número de variações por imagem original

# Augmentador robusto
augmentador = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=(0.6, 1.4),
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

def augmentar_classe(classe):
    in_dir = os.path.join(INPUT_DIR, classe)
    out_dir = os.path.join(OUTPUT_DIR, classe)
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(in_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        base = os.path.splitext(fname)[0]
        img_path = os.path.join(in_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        img = np.expand_dims(img, axis=-1)  # (H, W, 1)
        img = np.expand_dims(img, axis=0)   # (1, H, W, 1)

        # Salvar a imagem original também
        cv2.imwrite(os.path.join(out_dir, f"{base}_orig.png"), img[0, :, :, 0])

        # Gerar variações
        gen = augmentador.flow(img, batch_size=1)
        for i in range(COPIAS_EXTRA):
            aug = next(gen)[0].astype(np.uint8)
            out_name = f"{base}_aug{i+1}.png"
            cv2.imwrite(os.path.join(out_dir, out_name), aug[:, :, 0])

# Executar para todas as classes
for classe in CLASSES:
    augmentar_classe(classe)

print("✅ Augmentation completo. Imagens salvas em 'dataset/expanded/'")