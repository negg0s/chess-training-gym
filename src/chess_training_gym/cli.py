"""Command-line interface for chess-training-gym."""

import json
import os
import click

from .generate import generate_games, write_games
from .encode import fen_to_tensor, encode_position, batch_encode
from .eval import evaluate_model, build_synthetic_dataset
from .models import MockedModel
from .report import generate_report, write_report, read_results


@click.group()
def cli():
    """Chess Training Gym CLI."""


@cli.command()
@click.option("--num-games", "-n", default=10, type=int, help="Number of games.")
@click.option("--seed", default=0, type=int, help="Random seed.")
@click.option("--max-plies", default=80, type=int, help="Max plies per game.")
@click.option("--out", default="data/games.jsonl", help="Output file.")
def generate(num_games, seed, max_plies, out):
    """Generate synthetic games."""
    games = generate_games(num_games=num_games, seed=seed, max_plies=max_plies)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    write_games(out, games)
    click.echo(f"Generated {len(games)} games -> {out}")


@cli.command()
@click.option("--fen", default=None, help="Single FEN to encode.")
@click.option("--input", "inp", default=None, help="Input file (json/jsonl with FENs or games).")
@click.option("--out", default=None, help="Output file (json).")
def encode(fen, inp, out):
    """Encode positions (FEN -> tensors)."""
    fens = []
    if fen:
        fens.append(fen)
    if inp:
        with open(inp, "r", encoding="utf-8") as f:
            content = f.read().strip()
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        fens.append(item)
                    elif isinstance(item, dict):
                        if "fen" in item:
                            fens.append(item["fen"])
                        elif "positions" in item:
                            fens.extend(item["positions"])
            elif isinstance(data, dict) and "positions" in data:
                fens.extend(data["positions"])
        except json.JSONDecodeError:
            # jsonl
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, str):
                        fens.append(obj)
                    elif isinstance(obj, dict):
                        if "fen" in obj:
                            fens.append(obj["fen"])
                        elif "positions" in obj:
                            fens.extend(obj["positions"])
                except json.JSONDecodeError:
                    fens.append(line)
    if not fens:
        fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
    tensors = [encode_position(f) for f in fens]
    # make json serializable: convert tensor to list
    serializable = []
    for t in tensors:
        ten = t["tensor"]
        try:
            arr = ten.tolist()
        except Exception:
            arr = ten
        serializable.append({"fen": t["fen"], "shape": list(t["shape"]), "tensor": arr})
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(serializable, f)
        click.echo(f"Encoded {len(fens)} positions -> {out}")
    else:
        click.echo(json.dumps(serializable[0] if len(serializable) == 1 else serializable)[:500])


@cli.command()
@click.option("--num-positions", default=50, type=int, help="Synthetic positions.")
@click.option("--seed", default=0, type=int, help="Seed.")
@click.option("--k", default=3, type=int, help="Top-k.")
@click.option("--out", default="data/eval.json", help="Output results JSON.")
def eval(num_positions, seed, k, out):
    """Evaluate mocked model."""
    dataset = build_synthetic_dataset(num_positions=num_positions, seed=seed)
    model = MockedModel(seed=seed)
    results = evaluate_model(model, dataset, k=k)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    click.echo(f"Evaluated {results['num_samples']} samples -> {out}")
    click.echo(json.dumps(results, indent=2))


@cli.command()
@click.option("--input", "inp", required=True, help="Results JSON file.")
@click.option("--out", default="report.md", help="Output report file.")
@click.option("--format", "fmt", default="markdown", help="markdown or json.")
def report(inp, out, fmt):
    """Generate report from results."""
    results = read_results(inp)
    write_report(results, out, format=fmt)
    click.echo(f"Report written -> {out}")


def main():
    cli()


if __name__ == "__main__":
    main()