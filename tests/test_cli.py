from click.testing import CliRunner
from chess_training_gym.cli import cli
import json, os, tempfile


def test_generate_command():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "games.jsonl")
        res = runner.invoke(cli, ["generate", "-n", "2", "--seed", "1", "--out", out])
        assert res.exit_code == 0, res.output
        assert os.path.exists(out)


def test_encode_command():
    runner = CliRunner()
    res = runner.invoke(cli, ["encode", "--fen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"])
    assert res.exit_code == 0, res.output


def test_eval_command():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "eval.json")
        res = runner.invoke(cli, ["eval", "--num-positions", "5", "--seed", "1", "--out", out])
        assert res.exit_code == 0, res.output
        assert os.path.exists(out)


def test_report_command():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d:
        ev = os.path.join(d, "eval.json")
        rep = os.path.join(d, "report.md")
        runner.invoke(cli, ["eval", "--num-positions", "5", "--seed", "1", "--out", ev])
        res = runner.invoke(cli, ["report", "--input", ev, "--out", rep])
        assert res.exit_code == 0, res.output
        assert os.path.exists(rep)