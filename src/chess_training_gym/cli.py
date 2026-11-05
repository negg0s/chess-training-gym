"""Command-line interface for chess-training-gym."""

import json
import os

import click

from .board import START_FEN
from .dataset import compute_stats, read_jsonl, split_records, write_jsonl
from .encode import encode_position, read_positions
from .eval import (
    build_synthetic_dataset,
    evaluate_model,
    evaluate_predictions_file,
)
from .generate import generate_games, write_games
from .models import MockedModel
from .report import read_results, write_report


def _ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _tensor_as_list(tensor):
    if hasattr(tensor, "tolist"):
        return tensor.tolist()
    return tensor


def _serialize_position(position):
    return {
        "fen": position["fen"],
        "shape": list(position["shape"]),
        "tensor": _tensor_as_list(position["tensor"]),
    }


@click.group()
def cli():
    """Chess Training Gym CLI."""


@cli.command()
@click.option("--num-games", "-n", default=10, type=int, help="Number of games.")
@click.option("--seed", default=0, type=int, help="Random seed.")
@click.option("--max-plies", default=80, type=int, help="Max plies per game.")
@click.option("--out", default="data/games.jsonl", help="Output JSONL file.")
@click.option("--metadata/--no-metadata", default=True, help="Write a metadata header line.")
def generate(num_games, seed, max_plies, out, metadata):
    """Generate synthetic games as JSONL."""
    games = generate_games(num_games=num_games, seed=seed, max_plies=max_plies)
    _ensure_parent(out)
    header = None
    if metadata:
        header = {
            "generator": "chess-training-gym",
            "seed": seed,
            "num_games": len(games),
            "max_plies": max_plies,
        }
    write_games(out, games, metadata=header)
    click.echo(f"Generated {len(games)} games -> {out}")


@cli.command()
@click.option("--fen", multiple=True, help="FEN to encode; may be repeated.")
@click.option("--input", "inp", default=None, help="FEN file, JSON or JSONL input.")
@click.option("--out", default=None, help="Output file.")
@click.option("--format", "fmt", type=click.Choice(["json", "jsonl"]), default="json")
def encode(fen, inp, out, fmt):
    """Encode positions (FEN -> tensors)."""
    fens = list(fen)
    if inp:
        fens.extend(read_positions(inp))
    if not fens:
        fens = [START_FEN]
    encoded = [_serialize_position(encode_position(f)) for f in fens]
    if out:
        _ensure_parent(out)
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            if fmt == "jsonl":
                for item in encoded:
                    f.write(json.dumps(item) + "\n")
            else:
                json.dump(encoded, f)
        click.echo(f"Encoded {len(encoded)} positions -> {out}")
    elif len(encoded) == 1:
        click.echo(json.dumps(encoded[0]))
    else:
        click.echo(f"Encoded {len(encoded)} positions (use --out to save)")


@cli.command()
@click.option("--input", "inp", required=True, help="Games JSONL file.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead of text.")
@click.option("--out", default=None, help="Write stats JSON to this file.")
def stats(inp, as_json, out):
    """Summarize a JSONL dataset."""
    records = read_jsonl(inp)
    summary = compute_stats(records)
    if out:
        _ensure_parent(out)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    if as_json:
        click.echo(json.dumps(summary, indent=2))
        return
    click.echo(f"Games: {summary['games']}")
    positions = summary["positions"]
    click.echo(f"Positions: {positions if positions is not None else 'n/a'}")
    click.echo(f"Average game length: {summary['avg_game_length']:.1f} plies")
    if summary["games"] > 0 and summary["total_plies"] > 0:
        click.echo(f"Game length range: {summary['min_game_length']}-{summary['max_game_length']} plies")
    click.echo("Results:")
    for name, count in summary["result_distribution"].items():
        click.echo(f"  {name}: {count}")


@cli.command()
@click.option("--input", "inp", required=True, help="JSONL file to split.")
@click.option("--out-dir", default="data/splits", help="Directory for split files.")
@click.option("--train", default=0.8, type=float, help="Train ratio.")
@click.option("--val", default=0.1, type=float, help="Validation ratio.")
@click.option("--test", default=0.1, type=float, help="Test ratio.")
@click.option("--seed", default=0, type=int, help="Shuffle seed.")
@click.option("--shuffle/--no-shuffle", default=True, help="Shuffle before splitting.")
def split(inp, out_dir, train, val, test, seed, shuffle):
    """Split a JSONL dataset into train/val/test files."""
    records = read_jsonl(inp)
    try:
        parts = split_records(records, train, val, test, seed, shuffle)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    os.makedirs(out_dir, exist_ok=True)
    for name, rows in parts.items():
        path = os.path.join(out_dir, f"{name}.jsonl")
        write_jsonl(path, rows, metadata={
            "split": name,
            "seed": seed,
            "num_games": len(rows),
        })
        click.echo(f"{name}: {len(rows)} games -> {path}")


@cli.command()
@click.option("--input", "inp", required=True, help="Games JSONL file.")
@click.option("--out-dir", default="data/prepared", help="Directory for encoded split files.")
@click.option("--train", default=0.8, type=float, help="Train ratio.")
@click.option("--val", default=0.1, type=float, help="Validation ratio.")
@click.option("--test", default=0.1, type=float, help="Test ratio.")
@click.option("--seed", default=0, type=int, help="Shuffle seed.")
@click.option("--shuffle/--no-shuffle", default=True, help="Shuffle games before splitting.")
def prepare(inp, out_dir, train, val, test, seed, shuffle):
    """Split games, encode positions, and attach each next-move target."""
    games = read_jsonl(inp)
    for game_index, game in enumerate(games):
        if game.get("game_id") is None:
            game["game_id"] = game_index
    try:
        parts = split_records(games, train, val, test, seed, shuffle)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    os.makedirs(out_dir, exist_ok=True)
    for name, split_games in parts.items():
        samples = []
        for game in split_games:
            game_id = game["game_id"]
            positions = game.get("positions", [])
            moves = game.get("moves", [])
            if len(positions) != len(moves) + 1:
                raise click.ClickException(
                    "each game must have one more position than moves"
                )
            for ply, (fen, move) in enumerate(zip(positions, moves)):
                encoded = _serialize_position(encode_position(fen))
                samples.append({
                    "sample_id": f"{game_id}:{ply}",
                    "game_id": game_id,
                    "ply": ply,
                    "fen": fen,
                    "target_move": move,
                    "shape": encoded["shape"],
                    "tensor": encoded["tensor"],
                })
        path = os.path.join(out_dir, f"{name}.jsonl")
        write_jsonl(path, samples, metadata={
            "split": name,
            "seed": seed,
            "num_games": len(split_games),
            "num_samples": len(samples),
        })
        click.echo(f"{name}: {len(samples)} positions from {len(split_games)} games -> {path}")


@cli.command(name="eval")
@click.option("--predictions", default=None, help="Predictions JSONL file.")
@click.option("--labels", default=None, help="Optional labels JSONL file matched by FEN.")
@click.option("--num-positions", default=50, type=int, help="Synthetic positions.")
@click.option("--seed", default=0, type=int, help="Seed for the synthetic run.")
@click.option("--k", default=3, type=int, help="Top-k.")
@click.option("--out", default="data/eval.json", help="Output results JSON.")
def eval_cmd(predictions, labels, num_positions, seed, k, out):
    """Evaluate a model or a predictions file."""
    if predictions:
        results = evaluate_predictions_file(predictions, labels, k=k)
    else:
        dataset = build_synthetic_dataset(num_positions=num_positions, seed=seed)
        model = MockedModel(seed=seed)
        results = evaluate_model(model, dataset, k=k)
    _ensure_parent(out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    click.echo(f"Evaluated {results['num_samples']} samples -> {out}")
    click.echo(
        "top-1: {:.4f}  top-{}: {:.4f}  value pearson: {:.4f}".format(
            results["top_1_accuracy"],
            results["top_k"],
            results["top_k_accuracy"],
            results["value_correlation"],
        )
    )


@cli.command()
@click.option("--input", "inp", required=True, help="Results JSON file.")
@click.option("--out", default="report.md", help="Output report file.")
@click.option("--format", "fmt", type=click.Choice(["markdown", "json"]), default="markdown")
def report(inp, out, fmt):
    """Generate a report from results."""
    results = read_results(inp)
    _ensure_parent(out)
    write_report(results, out, format=fmt)
    click.echo(f"Report written -> {out}")





@cli.command("doctor")
@click.option("--json", "as_json", is_flag=True, help="Output diagnostics as JSON.")
def doctor(as_json):
    """Run environment and component diagnostics."""
    import sys
    from .board import START_FEN, legal_moves, parse_fen
    from .encode import encode_position

    checks = []
    checks.append(("python_version", True, sys.version.split()[0]))

    try:
        state = parse_fen(START_FEN)
        moves = list(legal_moves(state))
        checks.append(("board_engine", len(moves) == 20, f"startpos moves={len(moves)}"))
    except Exception as exc:
        checks.append(("board_engine", False, str(exc)))

    try:
        encoded = encode_position(START_FEN)
        shape = tuple(encoded.get("shape", ()))
        checks.append(("position_encoder", shape == (13, 8, 8), f"planes_shape={shape}"))
    except Exception as exc:
        checks.append(("position_encoder", False, str(exc)))

    try:
        import numpy as np
        checks.append(("numpy", True, f"version={np.__version__}"))
    except Exception as exc:
        checks.append(("numpy", False, str(exc)))

    passed = sum(1 for _, ok, _ in checks if ok)
    if as_json:
        payload = {
            "checks": [{"name": name, "ok": ok, "detail": detail} for name, ok, detail in checks],
            "passed": passed,
            "total": len(checks),
        }
        click.echo(json.dumps(payload, indent=2))
        if passed != len(checks):
            raise SystemExit(1)
        return

    for name, ok, detail in checks:
        status = "OK" if ok else "FAIL"
        click.echo(f"[{status}] {name:20} {detail}")
    click.echo(f"doctor: {passed}/{len(checks)} checks passed")
    if passed != len(checks):
        raise SystemExit(1)


def main():
    cli()


if __name__ == "__main__":
    main()
