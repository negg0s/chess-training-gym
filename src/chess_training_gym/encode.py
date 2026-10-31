"""Encode chess positions (FEN) into simple tensor-like structures."""

import json

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - depends on environment
    np = None
    _HAS_NUMPY = False

from .board import PIECE_PLANES, parse_fen

NUM_PLANES = 13
SIDE_PLANE = 12


def _empty_tensor():
    if _HAS_NUMPY:
        return np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    return [[[0.0 for _ in range(8)] for _ in range(8)] for _ in range(NUM_PLANES)]


def fen_to_tensor(fen):
    """Return a (13, 8, 8) tensor for a FEN string.

    Planes 0-5 hold white pawn, knight, bishop, rook, queen and king.
    Planes 6-11 hold the same pieces for black. Plane 12 is all ones when
    white is to move and all zeros otherwise.
    """
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
    if state["side"] == "w":
        if _HAS_NUMPY:
            tensor[SIDE_PLANE, :, :] = 1.0
        else:
            for r in range(8):
                for c in range(8):
                    tensor[SIDE_PLANE][r][c] = 1.0
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


def to_torch(value):
    """Convert a tensor-like value to a torch tensor.

    torch is optional; install it with ``pip install -e ".[torch]"``.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "torch is not installed; install the optional extra with "
            "'pip install -e \".[torch]\"'"
        ) from exc
    return torch.as_tensor(value)


def _extract_fens(obj):
    fens = []
    if isinstance(obj, str):
        fens.append(obj)
    elif isinstance(obj, list):
        for item in obj:
            fens.extend(_extract_fens(item))
    elif isinstance(obj, dict):
        if isinstance(obj.get("fen"), str):
            fens.append(obj["fen"])
        if isinstance(obj.get("positions"), list):
            fens.extend(_extract_fens(obj["positions"]))
    return fens


def read_positions(path):
    """Read positions from a FEN file or from JSON/JSONL.

    Supported inputs:
    - one FEN per line
    - a JSON array of FEN strings
    - a JSON object with ``positions``
    - JSONL records with ``fen`` or ``positions`` fields
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    try:
        return _extract_fens(json.loads(content))
    except json.JSONDecodeError:
        pass
    fens = []
    for lineno, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            if "/" not in line:
                raise ValueError(f"{path}:{lineno}: not a FEN or JSON record")
            fens.append(line)
        else:
            fens.extend(_extract_fens(obj))
    return fens


def encode_game(game):
    return [encode_position(fen) for fen in game.get("positions", [])]


# Backwards compatible alias.
encode = encode_position
