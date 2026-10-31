import json
from chess_training_gym.generate import generate_games, generate_game, write_games, read_games
import tempfile, os


def test_deterministic():
    a = generate_games(num_games=3, seed=42)
    b = generate_games(num_games=3, seed=42)
    assert a == b


def test_count_and_schema():
    games = generate_games(n=5, seed=1)
    assert len(games) == 5
    for g in games:
        assert "moves" in g
        assert "positions" in g
        assert len(g["positions"]) == g["num_plies"] + 1


def test_write_read_roundtrip():
    games = generate_games(num_games=2, seed=7)
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    write_games(path, games)
    loaded = read_games(path)
    assert loaded == games
    os.remove(path)