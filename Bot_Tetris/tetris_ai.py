# tetris_ai.py

import numpy as np

# Heurística baseada em Dellacherie, ajustável
WEIGHTS = {
    'aggregate_height': -0.510066,
    'complete_lines': 0.760666,
    'holes': -0.35663,
    'bumpiness': -0.184483
}

def get_column_heights(board):
    heights = np.zeros(board.shape[1], dtype=int)
    for j in range(board.shape[1]):
        column = board[:, j]
        filled = np.where(column == 1)[0]
        heights[j] = board.shape[0] - filled[0] if filled.size > 0 else 0
    return heights

def count_holes(board):
    holes = 0
    for j in range(board.shape[1]):
        column = board[:, j]
        filled = np.where(column == 1)[0]
        if filled.size > 0:
            holes += np.sum(column[filled[0]:] == 0)
    return holes

def get_bumpiness(heights):
    return np.sum(np.abs(np.diff(heights)))

def get_complete_lines(board):
    return np.sum(np.all(board == 1, axis=1))

def evaluate_board(board):
    heights = get_column_heights(board)
    agg_height = np.sum(heights)
    holes = count_holes(board)
    bumpiness = get_bumpiness(heights)
    complete_lines = get_complete_lines(board)

    score = (WEIGHTS['aggregate_height'] * agg_height +
             WEIGHTS['complete_lines'] * complete_lines +
             WEIGHTS['holes'] * holes +
             WEIGHTS['bumpiness'] * bumpiness)

    return score

def simulate_piece_drop(board, piece_type, rotation, column):
    # ⚠️ Simplificação: assume peça 1x1 para teste
    # Para peça real, implemente lógica de rotação e colisão conforme shape
    simulated = board.copy()
    for i in range(board.shape[0] - 1, -1, -1):
        if simulated[i, column] == 0:
            simulated[i, column] = 1
            break
    return simulated

def get_best_move(board, piece_type):
    best_score = -float('inf')
    best_rotation = 0
    best_col = 0

    possible_rotations = 4  # simplificação; defina por tipo real da peça

    for rotation in range(possible_rotations):
        for col in range(board.shape[1]):
            simulated = simulate_piece_drop(board, piece_type, rotation, col)
            score = evaluate_board(simulated)

            if score > best_score:
                best_score = score
                best_rotation = rotation
                best_col = col

    return best_rotation, best_col
