import cv2
import numpy as np
import mss
import time
import os

# Defina a região do jogo que contém a peça
PIECE_REGION = {
    'top': 150,
    'left': 885,
    'width': 265,
    'height': 515
}

# Inicialização do mss para captura da tela
sct = mss.mss()

# Configuração de saída para vídeos (opcional)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_out = cv2.VideoWriter("output.avi", fourcc, 20.0, (PIECE_REGION["width"], PIECE_REGION["height"]))

# Criação de diretório para salvar as imagens
DETECTED_PIECES_PATH = 'detected_pieces'
if not os.path.exists(DETECTED_PIECES_PATH):
    os.makedirs(DETECTED_PIECES_PATH)

# Inicialização das variáveis
prev_gray = None
last_cx, last_cy = None, None
frame_count = 0

print("Pressione ESC para sair...")

def classify_movement(dx, dy):
    """Classifica o movimento como horizontal ou vertical"""
    if abs(dy) > abs(dx):
        return "vertical"
    else:
        return "horizontal"

# Função para salvar a imagem detectada
def save_detected_piece(frame, frame_count, x, y, w, h):
    """Salva a parte da tela que contém a peça detectada"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    detected_piece = frame[y:y+h, x:x+w]
    image_path = os.path.join(DETECTED_PIECES_PATH, f"piece_{frame_count}_{timestamp}.png")
    cv2.imwrite(image_path, detected_piece)
    print(f"Imagem salva como: {image_path}")

while True:
    # Captura a tela da região definida (PIECE_REGION)
    screenshot = np.array(sct.grab(PIECE_REGION))
    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    if prev_gray is None:
        prev_gray = gray
        continue

    # Diferença entre os quadros
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Encontra os contornos das áreas com movimento
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 150:
            continue

        # Obtém as coordenadas do retângulo delimitador
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2

        # Calcula o movimento baseado na diferença entre os centros
        dx = dy = 0
        if last_cx is not None and last_cy is not None:
            dx = cx - last_cx
            dy = cy - last_cy
        last_cx, last_cy = cx, cy

        # Classifica o tipo de movimento
        movement = classify_movement(dx, dy)
        color = (0, 0, 255) if movement == "vertical" else (0, 255, 0)
        label = f"{movement.upper()}"

        # Desenha o retângulo ao redor da peça
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Salva a imagem da peça detectada
        save_detected_piece(frame, frame_count, x, y, w, h)

        # Exemplo de lógica de IA reativa (para movimentação)
        if movement == "vertical":
            print("IA: peça está descendo")
        else:
            print("IA: peça se movendo lateralmente")

    # Grava o quadro no vídeo
    video_out.write(frame)

    # Exibe o quadro com o retângulo de detecção
    cv2.imshow("Tetris - Detector", frame)

    # Atualiza o quadro anterior
    prev_gray = gray.copy()
    frame_count += 1

    # Aguarda a tecla ESC para sair
    if cv2.waitKey(1) == 27:  # ESC
        break

# Libera recursos
video_out.release()
cv2.destroyAllWindows()
