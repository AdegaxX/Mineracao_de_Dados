from piece_detector import detect_piece

if __name__ == "__main__":
    piece, _ = detect_piece(show_window=True)
    print("Peça detectada:", piece)
