import cv2
import numpy as np
from mss import mss
from config import BOARD_REGION

BOARD_HEIGHT = 20
BOARD_WIDTH = 10

def extract_board(debug=False):
    sct = mss()
    img = np.array(sct.grab(BOARD_REGION))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    cell_h = BOARD_REGION['height'] // BOARD_HEIGHT
    cell_w = BOARD_REGION['width'] // BOARD_WIDTH
    board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)

    for i in range(BOARD_HEIGHT):
        for j in range(BOARD_WIDTH):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell = binary[y1:y2, x1:x2]
            filled_ratio = np.sum(cell > 0) / (cell_h * cell_w)
            if filled_ratio > 0.2:
                board[i, j] = 1

    if debug:
        cv2.imshow("Tabuleiro", binary)
        cv2.waitKey(1)

    return board
