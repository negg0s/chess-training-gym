"""chess-training-gym: synthetic chess training data utilities."""

__version__ = "0.1.0"

from .generate import generate_games, generate_game, generate_synthetic_games
from .encode import fen_to_tensor, encode_position, encode_fen, batch_encode
from .eval import top_k_accuracy, value_correlation, evaluate_model, evaluate
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
    "top_k_accuracy",
    "value_correlation",
    "evaluate_model",
    "evaluate",
    "MockedModel",
    "generate_report",
    "write_report",
]