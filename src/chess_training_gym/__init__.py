"""chess-training-gym: synthetic chess training data utilities."""

__version__ = "0.1.0"

from .dataset import compute_stats, read_jsonl, split_records, write_jsonl
from .encode import (
    batch_encode,
    encode_fen,
    encode_position,
    fen_to_tensor,
    read_positions,
    to_torch,
)
from .eval import (
    evaluate,
    evaluate_model,
    evaluate_predictions,
    top_k_accuracy,
    value_correlation,
)
from .generate import generate_game, generate_games, generate_synthetic_games
from .models import MockedModel
from .report import generate_report, write_report

__all__ = [
    "generate_games",
    "generate_game",
    "generate_synthetic_games",
    "fen_to_tensor",
    "encode_position",
    "encode_fen",
    "batch_encode",
    "read_positions",
    "to_torch",
    "compute_stats",
    "split_records",
    "read_jsonl",
    "write_jsonl",
    "top_k_accuracy",
    "value_correlation",
    "evaluate_model",
    "evaluate_predictions",
    "evaluate",
    "MockedModel",
    "generate_report",
    "write_report",
]
