"""Synthetic chess game generation (deterministic given a seed)."""

import random

from .board import START_FEN, apply_uci, in_check, legal_moves, parse_fen, to_fen
from .dataset import read_jsonl, write_jsonl


def generate_game(seed, game_id=0, max_plies=80):
    rng = random.Random(int(seed) * 1000003 + int(game_id))
    state = parse_fen(START_FEN)
    moves = []
    positions = [to_fen(state)]
    result = "max_plies"
    for _ in range(max_plies):
        legal = legal_moves(state)
        if not legal:
            result = "checkmate" if in_check(state, state["side"]) else "stalemate"
            break
        move = rng.choice(legal)
        moves.append(move)
        state = apply_uci(state, move)
        positions.append(to_fen(state))
    return {
        "game_id": game_id,
        "seed": seed,
        "moves": moves,
        "positions": positions,
        "result": result,
        "num_plies": len(moves),
    }


def generate_games(num_games=10, seed=0, max_plies=80, **kwargs):
    n = kwargs.get("n", num_games)
    return [generate_game(seed, i, max_plies) for i in range(n)]


generate_synthetic_games = generate_games
generate = generate_games


def write_games(path, games, metadata=None):
    return write_jsonl(path, games, metadata=metadata)


def read_games(path):
    return read_jsonl(path)
