# chess-training-gym

Utilities for building small chess training datasets: synthetic game
generation, position encoding, dataset splitting, evaluation and reporting.
Pure Python core. `numpy` and `torch` are optional.

## Install

```bash
python -m pip install -e . --no-cache-dir
```

With optional dependencies:

```bash
python -m pip install -e ".[numpy]" --no-cache-dir
python -m pip install -e ".[torch]" --no-cache-dir
python -m pip install -e ".[dev]" --no-cache-dir
```

Python 3.9 or newer. The CLI is available as `chess-gym` and
`chess-training-gym`, and as `python -m chess_training_gym`.

## Commands

```bash
chess-gym --help
```

### generate

Generate random legal games. Output is JSONL, one game per line.

```bash
chess-gym generate --num-games 20 --seed 1 --max-plies 60 --out data/games.jsonl
```

A metadata line is written first unless `--no-metadata` is passed:

```json
{"generator": "chess-training-gym", "max_plies": 60, "num_games": 20, "seed": 1, "type": "metadata"}
```

Each game is deterministic for a given `seed` and game index.

### stats

```bash
chess-gym stats --input data/games.jsonl
chess-gym stats --input data/games.jsonl --json --out data/stats.json
```

Prints game count, position count, average game length and result
distribution.

### split

Split a JSONL file into train/val/test. Records are shuffled with `seed`
unless `--no-shuffle` is passed.

```bash
chess-gym split --input data/games.jsonl --out-dir data/splits \
  --train 0.8 --val 0.1 --test 0.1 --seed 7
```

Writes `train.jsonl`, `val.jsonl` and `test.jsonl`. Each file starts with a
metadata line that records the split name, seed and record count. The ratios
must sum to 1.0. Every input record goes to exactly one output file.

### encode

Encode FEN positions into a `(13, 8, 8)` tensor. Planes 0 to 5 are white
pawn, knight, bishop, rook, queen and king. Planes 6 to 11 are the black
pieces in the same order. Plane 12 is 1.0 when white is to move, 0.0
otherwise.

Single FEN:

```bash
chess-gym encode --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

JSONL input, one encoded record per line:

```bash
chess-gym encode --input data/games.jsonl --out data/encoded.jsonl --format jsonl
```

`--input` accepts:

- one FEN per line
- a JSON array of FEN strings
- a JSON object with a `positions` array
- JSONL where each record has `fen` or `positions`

### eval

Evaluate the built-in mock model on synthetic data:

```bash
chess-gym eval --num-positions 100 --seed 1 --k 3 --out data/eval.json
```

Evaluate a predictions file:

```bash
chess-gym eval --predictions data/predictions.jsonl \
  --labels data/labels.jsonl --k 3 --out data/eval.json
```

Labels are matched by `fen`. Without `--labels`, each prediction record must
contain its own target.

Output fields: `num_samples`, `top_1_accuracy`, `top_k`,
`top_k_accuracy`, `value_samples` and `value_correlation`.

### report

```bash
chess-gym report --input data/eval.json --out report.md
chess-gym report --input data/eval.json --out report.json --format json
```

## Data format

Game record:

```json
{
  "game_id": 0,
  "seed": 1,
  "moves": ["e2e4", "e7e5"],
  "positions": ["<start FEN>", "<FEN after e2e4>", "<FEN after e7e5>"],
  "result": "max_plies",
  "num_plies": 2
}
```

`result` is one of `checkmate`, `stalemate` or `max_plies`.
`len(positions) == num_plies + 1`. Moves use UCI notation. Promotions use a
fifth character, for example `a7a8q`.

Prediction record:

```json
{"fen": "<FEN>", "moves": ["e2e4", "d2d4"], "predicted_value": 0.2}
```

`moves` is ranked best first. `policy` and `top_moves` are also accepted.
`predicted_value` is optional.

Label record:

```json
{"fen": "<FEN>", "target_move": "e2e4", "value": 0.1}
```

`target`, `best_move`, `label` and `move` are accepted as target keys. Values
are compared only when both a prediction value and a label value are present.
Pearson correlation is 0.0 when it is undefined for the given sample.

## Python API

```python
from chess_training_gym import generate_games, fen_to_tensor, split_records, compute_stats

games = generate_games(num_games=5, seed=0, max_plies=40)
tensor = fen_to_tensor("8/8/8/8/8/8/8/K6k w - - 0 1")
parts = split_records(games, train=0.8, val=0.1, test=0.1, seed=0)
print(compute_stats(games))
```

`to_torch(tensor)` converts a tensor-like value when `torch` is installed.

## Limitations

- Games are random legal moves. They are not useful as training targets.
- `MockedModel` returns pseudo-random moves and values. It exists to test the
  evaluation pipeline, not to play chess.
- Move generation is plain Python and slow. Depth 3 perft from the start
  position is covered by tests, but this is not an engine.
- No repetition, fifty-move or insufficient material detection. Games stop on
  checkmate, stalemate or `max_plies`.
- No SAN, PGN, UCI engine protocol, opening book or tablebase support.
- Encoded positions have no move history or castling planes.
- Datasets are read into memory. Large files need a different reader.

## Development

```bash
python -m pip install -e ".[dev]" --no-cache-dir
pytest -q
python tools/check_text.py
```

`tools/check_text.py` fails on em dashes, en dashes and a list of filler
phrases. CI runs it before the test suite.
