"""JSONL dataset helpers: reading, writing, splitting and summary stats."""

import json
import random


def read_jsonl(path):
    """Read JSONL records, skipping blank lines and metadata headers."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if isinstance(obj, dict) and obj.get("type") == "metadata":
                continue
            records.append(obj)
    return records


def write_jsonl(path, records, metadata=None):
    """Write records as JSONL, optionally preceded by one metadata line."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        if metadata is not None:
            header = {"type": "metadata"}
            header.update(metadata)
            f.write(json.dumps(header, sort_keys=True) + "\n")
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def validate_ratios(train, val, test):
    for name, value in (("train", train), ("val", val), ("test", test)):
        if value < 0 or value > 1:
            raise ValueError(f"{name} ratio must be between 0 and 1")
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {total}")


def split_records(records, train=0.8, val=0.1, test=0.1, seed=0, shuffle=True):
    """Split records into train/val/test using ratio boundaries.

    The split is deterministic for a given seed. Every input record lands in
    exactly one output list.
    """
    validate_ratios(train, val, test)
    indices = list(range(len(records)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    ordered = [records[i] for i in indices]
    n = len(ordered)
    train_end = int(n * train)
    val_end = int(n * (train + val))
    return {
        "train": ordered[:train_end],
        "val": ordered[train_end:val_end],
        "test": ordered[val_end:],
    }


def game_length(record):
    """Return the number of plies in a game record, or None."""
    if isinstance(record.get("num_plies"), int):
        return record["num_plies"]
    if isinstance(record.get("moves"), list):
        return len(record["moves"])
    if isinstance(record.get("positions"), list) and record["positions"]:
        return len(record["positions"]) - 1
    return None


def compute_stats(records):
    """Summarize games: counts, result distribution and length statistics."""
    results = {}
    lengths = []
    positions = 0
    positions_known = False
    for record in records:
        result = record.get("result", "unknown")
        results[result] = results.get(result, 0) + 1
        length = game_length(record)
        if length is not None:
            lengths.append(length)
        if isinstance(record.get("positions"), list):
            positions += len(record["positions"])
            positions_known = True
    games = len(records)
    white_wins = results.get("1-0", 0)
    black_wins = results.get("0-1", 0)
    draws = results.get("1/2-1/2", 0)
    white_win_rate = (white_wins + 0.5 * draws) / games if games > 0 else 0.0
    black_win_rate = (black_wins + 0.5 * draws) / games if games > 0 else 0.0
    draw_rate = draws / games if games > 0 else 0.0

    return {
        "games": games,
        "positions": positions if positions_known else None,
        "avg_game_length": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "min_game_length": min(lengths) if lengths else 0,
        "max_game_length": max(lengths) if lengths else 0,
        "total_plies": sum(lengths),
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "white_win_rate": white_win_rate,
        "black_win_rate": black_win_rate,
        "draw_rate": draw_rate,
        "result_distribution": dict(sorted(results.items())),
    }


def calculate_win_rate(records, perspective="white"):
    """Compute score / win rate for white or black with zero-division safety.

    Draws contribute 0.5 points. Empty records return 0.0 without errors.
    """
    if not records:
        return 0.0
    stats = compute_stats(records) if isinstance(records, list) else records
    games = stats.get("games", 0)
    if games <= 0:
        return 0.0
    if perspective == "white":
        return stats.get("white_win_rate", 0.0)
    if perspective == "black":
        return stats.get("black_win_rate", 0.0)
    raise ValueError(f"perspective must be 'white' or 'black', got {perspective!r}")

