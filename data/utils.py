"""Data utility functions: board encoding, FEN conversion, orientation."""
from typing import Dict, List, Tuple

import numpy as np

# category_id → FEN character (uppercase=white, lowercase=black)
FEN_PIECE_MAP = {
    0: 'P', 1: 'R', 2: 'N', 3: 'B', 4: 'Q', 5: 'K',
    6: 'p', 7: 'r', 8: 'n', 9: 'b', 10: 'q', 11: 'k',
}

PIECE_NAME_TO_ID = {
    'white-pawn': 0, 'white-rook': 1, 'white-knight': 2,
    'white-bishop': 3, 'white-queen': 4, 'white-king': 5,
    'black-pawn': 6, 'black-rook': 7, 'black-knight': 8,
    'black-bishop': 9, 'black-queen': 10, 'black-king': 11,
    'empty': 12,
}

NUM_CLASSES = 13  # 12 piece types + empty


def chesspos_to_rowcol(pos: str) -> Tuple[int, int]:
    """'a8' → (row=0, col=0);  'h1' → (row=7, col=7)."""
    col = ord(pos[0]) - ord('a')  # file a-h → 0-7
    rank = int(pos[1]) - 1        # rank 1-8 → 0-7
    row = 7 - rank                # rank 8 → row 0, rank 1 → row 7
    return row, col


def annotations_to_board(pieces: List[Dict]) -> np.ndarray:
    """Convert a list of piece annotation dicts to an 8×8 int64 board.

    Empty squares are filled with class id 12.
    """
    board = np.full((8, 8), 12, dtype=np.int64)
    for piece in pieces:
        row, col = chesspos_to_rowcol(piece['chessboard_position'])
        board[row, col] = piece['category_id']
    return board


def board_to_fen(board: np.ndarray, active_color: str = 'w') -> str:
    """Convert an 8×8 board array to a FEN position string."""
    rows = []
    for row in range(8):
        empty = 0
        s = ''
        for col in range(8):
            cid = int(board[row, col])
            if cid == 12:
                empty += 1
            else:
                if empty:
                    s += str(empty)
                    empty = 0
                s += FEN_PIECE_MAP[cid]
        if empty:
            s += str(empty)
        rows.append(s)
    return '/'.join(rows) + f' {active_color} - - 0 1'


def normalize_corners(corners: Dict, width: int, height: int) -> np.ndarray:
    """Normalize corner pixel coordinates to [0, 1].

    Returns shape (8,) in order: top_left_x, top_left_y, top_right_x, ...
    """
    order = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
    coords = []
    for key in order:
        x, y = corners[key]
        coords.extend([x / width, y / height])
    return np.array(coords, dtype=np.float32)


def compute_orientation(pieces: List[Dict]) -> int:
    """Derive board orientation (0-3) from piece positions and image bboxes.

    Orientation encodes which direction files (a→h) and ranks (1→8) run:
      0: files left→right,  ranks bottom→top  (standard white perspective)
      1: files right→left,  ranks bottom→top
      2: files left→right,  ranks top→bottom
      3: files right→left,  ranks top→bottom

    Returns 0 when fewer than 4 pieces have bbox annotations.
    """
    # Only ~19% of pieces carry bbox; images without them get orientation=0
    bbox_pieces = [p for p in pieces if 'bbox' in p]
    if len(bbox_pieces) < 4:
        return 0
    pieces = bbox_pieces

    files = np.array([ord(p['chessboard_position'][0]) - ord('a') for p in pieces], float)
    ranks = np.array([int(p['chessboard_position'][1]) - 1 for p in pieces], float)
    xs = np.array([p['bbox'][0] + p['bbox'][2] / 2 for p in pieces], float)
    ys = np.array([p['bbox'][1] + p['bbox'][3] / 2 for p in pieces], float)

    def safe_corr(a, b):
        if np.std(a) < 1e-6 or np.std(b) < 1e-6:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    file_lr = safe_corr(files, xs) > 0   # True: a is left of h
    rank_bt = safe_corr(ranks, ys) < 0   # True: rank 1 is below rank 8 (standard)

    return (0 if (file_lr and rank_bt)
            else 1 if (not file_lr and rank_bt)
            else 2 if (file_lr and not rank_bt)
            else 3)
