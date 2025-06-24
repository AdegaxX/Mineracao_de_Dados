import mss
import cv2
import numpy as np

# Valores iniciais estimados
top = 150
left = 880
width = 275
height = 510

print("[↑ ↓ ← →] movem a região")
print("[W/S] ajustam altura | [A/D] ajustam largura")
print("[ESC] para sair\n")

with mss.mss() as sct:
    while True:
        monitor = {"top": top, "left": left, "width": width, "height": height}
        frame = np.array(sct.grab(monitor))

        display = cv2.resize(frame, (300, 600), interpolation=cv2.INTER_NEAREST)
        text = f"T:{top} L:{left} W:{width} H:{height}"
        cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Ajustar Região da Peça", display)

        key = cv2.waitKey(100)

        if key == 27:  # ESC
            break
        elif key == ord('w'):
            height = max(50, height - 5)
        elif key == ord('s'):
            height += 5
        elif key == ord('a'):
            width = max(50, width - 5)
        elif key == ord('d'):
            width += 5
        elif key == ord('i'):  # ↑
            top = max(0, top - 5)
        elif key == ord('k'):  # ↓
            top += 5
        elif key == ord('j'):  # ←
            left = max(0, left - 5)
        elif key == ord('l'):  # →
            left += 5

cv2.destroyAllWindows()
print(f"\nVALORES FINAIS PARA PIECE_REGION:")
print(f"PIECE_REGION = {{'top': {top}, 'left': {left}, 'width': {width}, 'height': {height}}}")