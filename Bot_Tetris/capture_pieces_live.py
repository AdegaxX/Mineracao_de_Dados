import cv2
import os
import uuid
import time
import numpy as np
import mss
import keyboard

# Define a região da peça no jogo (ajuste conforme necessário)
PIECE_REGION = {'top': 150, 'left': 885, 'width': 265, 'height': 515}
SAVE_DIR = "dataset/manual"

# Cria as pastas, se necessário
CLASSES = ['I', 'O', 'T', 'S', 'Z', 'L', 'J']
for cls in CLASSES:
    os.makedirs(os.path.join(SAVE_DIR, cls), exist_ok=True)

print("🕹️ Pressione uma das teclas [I, O, T, S, Z, L, J] para salvar a imagem capturada como a peça correspondente.")
print("❌ Pressione 'ESC' para sair.")

with mss.mss() as sct:
    while True:
        # Captura a imagem da região definida
        img = np.array(sct.grab(PIECE_REGION))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (40, 40), interpolation=cv2.INTER_AREA)

        # Mostra a imagem para você confirmar
        cv2.imshow("Peça atual", resized)

        # Espera tecla de 50ms
        key = cv2.waitKey(50)

        for cls in CLASSES:
            if keyboard.is_pressed(cls.lower()) or keyboard.is_pressed(cls.upper()):
                filename = f"{cls}_{uuid.uuid4().hex[:8]}.png"
                path = os.path.join(SAVE_DIR, cls, filename)
                cv2.imwrite(path, resized)
                print(f"💾 {filename} salvo em {cls}/")
                time.sleep(0.3)  # Anti-repique

        if keyboard.is_pressed("esc"):
            print("❎ Encerrando a captura...")
            break

cv2.destroyAllWindows()
