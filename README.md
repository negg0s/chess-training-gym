# Chess Training Gym

Chess Training Gym is a small data-preparation and evaluation toolkit for
people who already have chess games and a model they want to train or test.
It converts game records into encoded board positions with next-move targets,
keeps all positions from a game in one split, and scores predictions from your
own model.

It is not a chess engine and it does not train a neural network. Bring your
own games, training loop, model and inference code.

## Try the complete example

The repository includes four short, legal opening lines and a fixed prediction
file. This workflow summarizes the source data, creates encoded train,
validation and test files, then evaluates predictions against labels from the
held-out test split. It runs without NumPy or PyTorch.

Install the project once:

```console
python -m pip install -e .
```

Run this from the repository root:

```console
python -m chess_training_gym stats --input examples/tiny_games.jsonl --json
python -m chess_training_gym prepare --input examples/tiny_games.jsonl --out-dir .tmp/tiny-prepared --train 0.5 --val 0 --test 0.5 --seed 17
python -m chess_training_gym eval --predictions examples/tiny_demo_predictions.jsonl --labels .tmp/tiny-prepared/test.jsonl --k 3 --out .tmp/tiny-eval.json
```

`prepare` performs the split and encoding together. It writes `train.jsonl`,
`val.jsonl` and `test.jsonl` into `.tmp/tiny-prepared`. Each data row includes
a stable `sample_id`, game ID, ply, FEN, next move target, tensor shape and
the encoded tensor. The split is done by whole games, so a position from one
game cannot leak into another split. The example uses no validation games to
keep the sample small. Use a nonzero validation ratio for model selection.

The example predictions are fixed output from the included `MockedModel`,
provided to demonstrate file formats and metric calculation. They are not
real model output. The opening lines are not a useful training corpus.
Evaluation reports top-1 and top-k move accuracy. Value correlation is 0.0
because these samples have no engine evaluation labels.

## Use your games and model

### 1. Export your games

Provide JSONL, one game per line. Every game needs a `moves` list of UCI moves
and a `positions` list containing the FEN before each move plus the final FEN.
There must be exactly one more position than moves. For example:

```json
{"game_id": "game-001", "moves": ["e2e4", "e7e5"], "positions": ["<FEN before e2e4>", "<FEN before e7e5>", "<FEN after e7e5>"]}
```

FEN strings must be valid. UCI promotions include the promotion piece, such as
`a7a8q`. Your data should contain positions paired with meaningful moves. The
synthetic `generate` command is useful for testing file handling, not for
making chess training data.

### 2. Check and prepare it

```console
chess-gym stats --input data/games.jsonl --json --out data/stats.json
chess-gym prepare --input data/games.jsonl --out-dir data/prepared --train 0.8 --val 0.1 --test 0.1 --seed 7
```

The prepared files are JSONL. Each row has `fen`, `target_move`, and `tensor`
with shape `(13, 8, 8)`. Planes 0 through 5 represent white pawn, knight,
bishop, rook, queen and king. Planes 6 through 11 represent the same black
pieces. Plane 12 is all ones when white is to move and all zeros when black is
to move. Castling rights, en passant, move counters and move history are not
encoded. The FEN is preserved alongside the tensor.

Use the train rows in your training loop and the validation rows for model
selection. The tool only prepares examples. Your training loop decides the
loss, optimizer, batching and checkpoints.

### 3. Run your model on test rows

Your inference adapter reads `data/prepared/test.jsonl` and writes one JSONL
prediction per input row. Keep `sample_id` and `fen` from the input. `moves`
must be your ranked UCI move guesses, best first. `predicted_value` is optional.

```json
{"sample_id": "game-001:0", "fen": "<same FEN>", "moves": ["e2e4", "d2d4"], "predicted_value": 0.2}
```

When the model has a value head, add matching `value` fields to your original
game-position labels or supply labels with `sample_id`, `fen`, `target_move`
and `value`. The `prepare` output already has move labels and matching sample
IDs. Evaluate the predictions against that held-out split:

```console
chess-gym eval --predictions data/my-predictions.jsonl --labels data/prepared/test.jsonl --k 3 --out data/eval.json
chess-gym report --input data/eval.json --out data/eval-report.md
```

Labels are matched by `sample_id` when present, otherwise by FEN. Stable IDs
matter when the same FEN appears more than once with different recorded
moves. The evaluator does not check whether model moves are legal.

## Other commands

Generate random legal games for plumbing tests:

```console
chess-gym generate --num-games 20 --seed 1 --max-plies 60 --out data/random-games.jsonl
```

Split raw game records without encoding:

```console
chess-gym split --input data/games.jsonl --out-dir data/splits --train 0.8 --val 0.1 --test 0.1 --seed 7
```

Encode positions from an existing FEN, JSON or JSONL file:

```console
chess-gym encode --input data/games.jsonl --out data/encoded.jsonl --format jsonl
```

The Python API exposes `generate_games`, `fen_to_tensor`, `split_records`,
`compute_stats`, `evaluate_predictions` and related helpers. Install the
optional `numpy` extra for NumPy arrays or `torch` extra for `to_torch`.

## Scope and limits

- This project prepares data and scores predictions. It does not train or
  save a model.
- Randomly generated games are legal move sequences, not sensible chess play
  or good training targets.
- Move accuracy measures agreement with recorded moves, not chess strength.
- Value correlation requires value labels and predictions. It is 0.0 when
  there are too few samples or either side has no variance.
- The included board code is a small pure Python implementation, not an
  engine. It does not detect repetition, the fifty-move rule or insufficient
  material, and does not support PGN or UCI engine communication.
- Data is read into memory. Use a different pipeline for very large corpora.

## Development

```console
python -m pip install -e ".[dev]"
pytest -q
python tools/check_text.py
```
