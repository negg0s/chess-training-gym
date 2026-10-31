"""Synthetic chess game generation (deterministic given a seed)."""

import json
import random
from .board import START_FEN, parse_fen, to_fen, legal_moves, apply_uci


def generate_game(seed, game_id=0, max_plies=80):
    rng = random.Random(int(seed) * 1000003 + int(game_id))
    state = parse_fen(START_FEN)
    moves = []
    positions = [to_fen(state)]
    result = "max_plies"
    for _ in range(max_plies):
        ms = legal_moves(state)
        if not ms:
            result = "checkmate" if _in_check_state(state) else "stalemate"
            break
        move = rng.choice(ms)
        moves.append(move)
        state = apply_uci(state, move)
        positions.append(to_fen(state))
    else:
        result = "max_plies"
    return {
        "game_id": game_id,
        "seed": seed,
        "moves": moves,
        "positions": positions,
        "result": result,
        "num_plies": len(moves),
    }


def _in_check_state(state):
    # local import to avoid circular
    from .board import in_check
    return in_check(state, state["side"])


def generate_games(num_games=10, seed=0, max_plies=80, **kwargs):
    n = kwargs.get("n", num_games)
    games = [generate_game(seed, i, max_plies) for i in range(n)]
    return games


# aliases
generate_synthetic_games = generate_games
generate = generate_games


def write_games(path, games):
    with open(path, "w", encoding="utf-8") as f:
        for g in games:
            f.write(json.dumps(g) + "\n")
    return path


def read_games(path):
    games = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))
    return games