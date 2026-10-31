"""Encode chess positions (FEN) into simple tensor-like structures."""

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None
    _HAS_NUMPY = False

from .board import parse_fen, PIECE_PLANES

NUM_PLANES = 13


def _empty_tensor():
    if _HAS_NUMPY:
        return np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    return [[[0.0 for _ in range(8)] for _ in range(8)] for _ in range(NUM_PLANES)]


def fen_to_tensor(fen):
    """Return a (13, 8, 8) tensor for a FEN string."""
    state = parse_fen(fen)
    board = state["board"]
    tensor = _empty_tensor()
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece in PIECE_PLANES:
                plane = PIECE_PLANES[piece]
                if _HAS_NUMPY:
                    tensor[plane, r, c] = 1.0
                else:
                    tensor[plane][r][c] = 1.0
    side_plane = 12
    if state["side"] == "w":
        if _HAS_NUMPY:
            tensor[side_plane, :, :] = 1.0
        else:
            for r in range(8):
                for c in range(8):
                    tensor[side_plane][r][c] = 1.0
    if _HAS_NUMPY:
        return tensor
    return tensor


def encode_position(fen):
    return {
        "fen": fen,
        "tensor": fen_to_tensor(fen),
        "shape": (NUM_PLANES, 8, 8),
    }


def encode_fen(fen):
    return encode_position(fen)


def encode_positions(fens):
    return [fen_to_tensor(f) for f in fens]


def batch_encode(fens):
    tensors = encode_positions(fens)
    if _HAS_NUMPY:
        return np.stack(tensors, axis=0)
    return tensors


# aliases
encode = encode_position
encode_game_positions = None


def encode_game(game):
    return [encode_position(f) for f in game.get("positions", [])]