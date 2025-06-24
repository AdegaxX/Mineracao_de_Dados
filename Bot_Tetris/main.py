import time
import threading
import pyautogui
from piece_detector import detect_piece
from board_extractor import extract_board
from tetris_ai import get_best_move

def perform_move(rotation, column):
    for _ in range(rotation):
        pyautogui.press('up')
    pyautogui.press('space')
    pyautogui.press('left', presses=10)
    pyautogui.press('right', presses=column)

def game_loop():
    while True:
        piece_type, _, confidence = detect_piece(show_window=True)
        print(f"Peça detectada: {piece_type} (confiança: {confidence:.2f})")

        board = extract_board()
        rotation, col = get_best_move(board, piece_type)
        print(f"Jogada: rot={rotation}, col={col}")

        perform_move(rotation, col)
        time.sleep(1)

if __name__ == "__main__":
    print("Prepare o jogo... iniciando em 5 segundos")
    time.sleep(5)
    threading.Thread(target=game_loop).start()
