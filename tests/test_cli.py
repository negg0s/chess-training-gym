import json
from pathlib import Path

from click.testing import CliRunner

from chess_training_gym.cli import cli

FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def run(args):
    return CliRunner().invoke(cli, args)


def test_help():
    result = run(["--help"])
    assert result.exit_code == 0, result.output
    for command in ("generate", "encode", "split", "stats", "eval", "report"):
        assert command in result.output


def test_generate_command(tmp_path):
    out = tmp_path / "games.jsonl"
    result = run(["generate", "-n", "2", "--seed", "1", "--out", str(out)])
    assert result.exit_code == 0, result.output
    lines = out.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["type"] == "metadata"
    assert json.loads(lines[0])["seed"] == 1
    assert len(lines) == 3


def test_generate_no_metadata(tmp_path):
    out = tmp_path / "games.jsonl"
    result = run(["generate", "-n", "1", "--out", str(out), "--no-metadata"])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text(encoding="utf-8"))["game_id"] == 0


def test_encode_fen(capsys):
    result = run(["encode", "--fen", FEN])
    assert result.exit_code == 0, result.output
    assert FEN in result.output


def test_encode_jsonl_input(tmp_path):
    src = tmp_path / "games.jsonl"
    src.write_text(
        '{"positions": ["%s"]}\n' % FEN,
        encoding="utf-8",
    )
    out = tmp_path / "encoded.jsonl"
    result = run(["encode", "--input", str(src), "--out", str(out), "--format", "jsonl"])
    assert result.exit_code == 0, result.output
    record = json.loads(out.read_text(encoding="utf-8").strip())
    assert record["fen"] == FEN
    assert record["shape"] == [13, 8, 8]


def test_stats_command(tmp_path):
    games = tmp_path / "games.jsonl"
    run(["generate", "-n", "3", "--seed", "2", "--max-plies", "10", "--out", str(games)])
    result = run(["stats", "--input", str(games)])
    assert result.exit_code == 0, result.output
    assert "Games: 3" in result.output
    assert "Positions:" in result.output
    assert "Average game length:" in result.output


def test_stats_json(tmp_path):
    games = tmp_path / "games.jsonl"
    run(["generate", "-n", "2", "--seed", "2", "--max-plies", "5", "--out", str(games)])
    out = tmp_path / "stats.json"
    result = run(["stats", "--input", str(games), "--json", "--out", str(out)])
    assert result.exit_code == 0, result.output
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["games"] == 2
    assert summary["positions"] == 12


def test_split_command(tmp_path):
    games = tmp_path / "games.jsonl"
    run(["generate", "-n", "10", "--seed", "4", "--max-plies", "6", "--out", str(games)])
    out_dir = tmp_path / "splits"
    result = run([
        "split", "--input", str(games), "--out-dir", str(out_dir),
        "--train", "0.6", "--val", "0.2", "--test", "0.2", "--seed", "7",
    ])
    assert result.exit_code == 0, result.output
    counts = []
    for name in ("train", "val", "test"):
        path = out_dir / f"{name}.jsonl"
        assert path.exists()
        lines = path.read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[0])["split"] == name
        counts.append(len(lines) - 1)
    assert counts == [6, 2, 2]


def test_split_bad_ratios(tmp_path):
    games = tmp_path / "games.jsonl"
    games.write_text('{"game_id": 0}\n', encoding="utf-8")
    result = run(["split", "--input", str(games), "--train", "0.5", "--val", "0.1", "--test", "0.1"])
    assert result.exit_code != 0


def test_prepare_creates_encoded_move_targets_without_game_leakage(tmp_path):
    games = tmp_path / "games.jsonl"
    run(["generate", "-n", "10", "--seed", "4", "--max-plies", "6", "--out", str(games)])
    out_dir = tmp_path / "prepared"

    result = run([
        "prepare", "--input", str(games), "--out-dir", str(out_dir),
        "--train", "0.6", "--val", "0.2", "--test", "0.2", "--seed", "7",
    ])

    assert result.exit_code == 0, result.output
    records = {}
    for split_name in ("train", "val", "test"):
        path = out_dir / f"{split_name}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert rows[0]["split"] == split_name
        records[split_name] = rows[1:]
        assert records[split_name]
        assert all(row["target_move"] and row["fen"] for row in records[split_name])
        assert all(row["shape"] == [13, 8, 8] for row in records[split_name])

    game_ids = {name: {row["game_id"] for row in rows} for name, rows in records.items()}
    assert game_ids["train"].isdisjoint(game_ids["val"])
    assert game_ids["train"].isdisjoint(game_ids["test"])
    assert game_ids["val"].isdisjoint(game_ids["test"])
    assert sum(map(len, records.values())) == 60


def test_tiny_example_runs_through_stats_prepare_and_eval(tmp_path):
    examples = Path(__file__).parents[1] / "examples"
    games = examples / "tiny_games.jsonl"
    predictions = examples / "tiny_demo_predictions.jsonl"
    out_dir = tmp_path / "prepared"

    stats = run(["stats", "--input", str(games), "--json"])
    assert stats.exit_code == 0, stats.output
    assert json.loads(stats.output)["games"] == 4

    prepared = run([
        "prepare", "--input", str(games), "--out-dir", str(out_dir),
        "--train", "0.5", "--val", "0", "--test", "0.5", "--seed", "17",
    ])
    assert prepared.exit_code == 0, prepared.output

    result_file = tmp_path / "eval.json"
    evaluated = run([
        "eval", "--predictions", str(predictions),
        "--labels", str(out_dir / "test.jsonl"), "--k", "3",
        "--out", str(result_file),
    ])
    assert evaluated.exit_code == 0, evaluated.output
    result = json.loads(result_file.read_text(encoding="utf-8"))
    assert result["num_samples"] == 12
    assert result["top_k"] == 3
    assert result["top_k_accuracy"] >= result["top_1_accuracy"]


def test_eval_command(tmp_path):
    out = tmp_path / "eval.json"
    result = run(["eval", "--num-positions", "5", "--seed", "1", "--out", str(out)])
    assert result.exit_code == 0, result.output
    results = json.loads(out.read_text(encoding="utf-8"))
    assert results["num_samples"] == 5
    assert "top_1_accuracy" in results


def test_eval_predictions_file(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    labels = tmp_path / "labels.jsonl"
    predictions.write_text(
        '{"fen": "a", "moves": ["e2e4", "d2d4"], "predicted_value": 0.2}\n'
        '{"fen": "b", "moves": ["c2c4"], "predicted_value": -0.2}\n',
        encoding="utf-8",
    )
    labels.write_text(
        '{"fen": "a", "target_move": "e2e4", "value": 0.1}\n'
        '{"fen": "b", "target_move": "c2c4", "value": -0.1}\n',
        encoding="utf-8",
    )
    out = tmp_path / "eval.json"
    result = run([
        "eval", "--predictions", str(predictions), "--labels", str(labels),
        "--k", "2", "--out", str(out),
    ])
    assert result.exit_code == 0, result.output
    results = json.loads(out.read_text(encoding="utf-8"))
    assert results["num_samples"] == 2
    assert results["top_1_accuracy"] == 1.0
    assert results["top_k_accuracy"] == 1.0
    assert results["value_correlation"] > 0.99


def test_report_command(tmp_path):
    ev = tmp_path / "eval.json"
    rep = tmp_path / "report.md"
    run(["eval", "--num-positions", "5", "--seed", "1", "--out", str(ev)])
    result = run(["report", "--input", str(ev), "--out", str(rep)])
    assert result.exit_code == 0, result.output
    assert rep.exists()
