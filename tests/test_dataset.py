import json

from chess_training_gym.dataset import (
    compute_stats,
    read_jsonl,
    split_records,
    write_jsonl,
)


def make_records(n):
    return [
        {
            "game_id": i,
            "moves": ["e2e4"] * (i + 1),
            "num_plies": i + 1,
            "positions": ["fen"] * (i + 2),
            "result": "max_plies" if i % 2 else "checkmate",
        }
        for i in range(n)
    ]


def test_split_counts_and_seed():
    records = make_records(100)
    first = split_records(records, 0.8, 0.1, 0.1, seed=3)
    second = split_records(records, 0.8, 0.1, 0.1, seed=3)
    assert first == second
    assert len(first["train"]) == 80
    assert len(first["val"]) == 10
    assert len(first["test"]) == 10


def test_split_keeps_every_record():
    records = make_records(37)
    parts = split_records(records, 0.7, 0.2, 0.1, seed=11)
    ids = [r["game_id"] for part in parts.values() for r in part]
    assert sorted(ids) == [r["game_id"] for r in records]


def test_split_without_shuffle_keeps_order():
    records = make_records(10)
    parts = split_records(records, 0.5, 0.25, 0.25, shuffle=False)
    assert [r["game_id"] for r in parts["train"]] == [0, 1, 2, 3, 4]
    assert [r["game_id"] for r in parts["val"]] == [5, 6]
    assert [r["game_id"] for r in parts["test"]] == [7, 8, 9]


def test_invalid_ratios():
    try:
        split_records(make_records(5), 0.5, 0.2, 0.1)
    except ValueError:
        return
    raise AssertionError("invalid ratios were accepted")


def test_stats():
    summary = compute_stats(make_records(4))
    assert summary["games"] == 4
    assert summary["positions"] == 2 + 3 + 4 + 5
    assert summary["avg_game_length"] == 2.5
    assert summary["result_distribution"] == {"checkmate": 2, "max_plies": 2}


def test_jsonl_roundtrip_skips_metadata(tmp_path):
    path = tmp_path / "games.jsonl"
    records = make_records(2)
    write_jsonl(path, records, metadata={"seed": 1, "num_games": 2})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["type"] == "metadata"
    assert read_jsonl(path) == records
