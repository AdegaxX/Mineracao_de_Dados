import numpy as np

BOARD_HEIGHT = 20
BOARD_WIDTH = 10

PIECES = {
    'I': [np.array([[1, 1, 1, 1]]), np.array([[1], [1], [1], [1]])],
    'O': [np.array([[1, 1], [1, 1]])],
    'T': [np.array([[0, 1, 0], [1, 1, 1]]),
          np.array([[1, 0], [1, 1], [1, 0]]),
          np.array([[1, 1, 1], [0, 1, 0]]),
          np.array([[0, 1], [1, 1], [0, 1]])],
    'S': [np.array([[0, 1, 1], [1, 1, 0]]),
          np.array([[1, 0], [1, 1], [0, 1]])],
    'Z': [np.array([[1, 1, 0], [0, 1, 1]]),
          np.array([[0, 1], [1, 1], [1, 0]])],
    'L': [np.array([[1, 0], [1, 0], [1, 1]]),
          np.array([[1, 1, 1], [1, 0, 0]]),
          np.array([[1, 1], [0, 1], [0, 1]]),
          np.array([[0, 0, 1], [1, 1, 1]])],
    'J': [np.array([[0, 1], [0, 1], [1, 1]]),
          np.array([[1, 0, 0], [1, 1, 1]]),
          np.array([[1, 1], [1, 0], [1, 0]]),
          np.array([[1, 1, 1], [0, 0, 1]])]
}

WEIGHTS = {
    "lines": 0.8,
    "holes": -0.7,
    "height": -0.5
}

def get_holes(board):
    holes = 0
    for col in range(board.shape[1]):
        col_data = board[:, col]
        seen_block = False
        for cell in col_data:
            if cell:
                seen_block = True
            elif seen_block:
                holes += 1
    return holes

def get_height(board):
    height = 0
    for col in range(board.shape[1]):
        cells = board[:, col]
        top = np.argmax(cells) if np.any(cells) else 0
        height += (BOARD_HEIGHT - top) if np.any(cells) else 0
    return height

def count_complete_lines(board):
    return np.sum(np.all(board == 1, axis=1))

def simulate(board, piece, rotation, col):
    piece_matrix = PIECES[piece][rotation]
    h, w = piece_matrix.shape
    sim_board = board.copy()
    for row in range(BOARD_HEIGHT - h + 1):
        if np.any(sim_board[row:row + h, col:col + w] & piece_matrix):
            break
    else:
        row = BOARD_HEIGHT - h

    row -= 1
    while row >= 0 and not np.any(sim_board[row:row + h, col:col + w] & piece_matrix):
        row -= 1
    row += 1
    if row + h > BOARD_HEIGHT:
        return board  # invalid
    sim_board[row:row + h, col:col + w] |= piece_matrix
    return sim_board

def get_best_move(board, piece):
    best_score = -np.inf
    best_move = (0, 0)

    for r, rot in enumerate(PIECES[piece]):
        h, w = rot.shape
        for col in range(BOARD_WIDTH - w + 1):
            sim_board = simulate(board, piece, r, col)
            if sim_board is board:
                continue

            score = (
                WEIGHTS["lines"] * count_complete_lines(sim_board) +
                WEIGHTS["holes"] * get_holes(sim_board) +
                WEIGHTS["height"] * get_height(sim_board)
            )
            if score > best_score:
                best_score = score
                best_move = (r, col)
    return best_move
