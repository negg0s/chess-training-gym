from chess_training_gym.eval import (
    build_synthetic_dataset,
    evaluate_model,
    evaluate_predictions,
    top_k_accuracy,
    value_correlation,
)
from chess_training_gym.models import MockedModel


def test_topk():
    preds = [["a", "b", "c"], ["x", "y"]]
    targets = ["b", "z"]
    assert top_k_accuracy(preds, targets, k=2) == 0.5


def test_corr_perfect():
    assert abs(value_correlation([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-9


def test_eval_runs():
    ds = build_synthetic_dataset(num_positions=10, seed=1)
    model = MockedModel(seed=1, perfect=True)
    # Make perfect model actually perfect by overriding target via dataset? Use random model instead just runs
    res = evaluate_model(MockedModel(seed=1), ds, k=2)
    assert "top_k_accuracy" in res
    assert "value_correlation" in res
    assert res["num_samples"] == 10


def test_evaluate_predictions_with_labels():
    predictions = [
        {"fen": "a", "moves": ["e2e4", "d2d4"], "predicted_value": 0.5},
        {"fen": "b", "moves": ["c2c4", "g1f3"], "predicted_value": -0.5},
        {"fen": "c", "moves": ["d2d4"], "predicted_value": 0.1},
    ]
    labels = [
        {"fen": "a", "target_move": "d2d4", "value": 0.4},
        {"fen": "b", "target_move": "c2c4", "value": -0.4},
        {"fen": "c", "target_move": "e2e4", "value": 0.0},
    ]
    res = evaluate_predictions(predictions, labels=labels, k=2)
    assert res["num_samples"] == 3
    assert res["top_1_accuracy"] == 1 / 3
    assert res["top_k_accuracy"] == 2 / 3
    assert res["value_samples"] == 3
    assert res["value_correlation"] > 0.99


def test_evaluate_predictions_self_contained():
    predictions = [
        {"moves": [["e2e4", 0.9], ["d2d4", 0.1]], "target_move": "e2e4"},
        {"moves": ["a2a3"], "target_move": "b2b3"},
    ]
    res = evaluate_predictions(predictions, k=1)
    assert res["top_1_accuracy"] == 0.5
    assert res["value_samples"] == 0


def test_evaluate_predictions_matches_sample_ids_when_fens_repeat():
    predictions = [
        {"sample_id": "g1:0", "fen": "same", "moves": ["e2e4"]},
        {"sample_id": "g2:0", "fen": "same", "moves": ["d2d4"]},
    ]
    labels = [
        {"sample_id": "g1:0", "fen": "same", "target_move": "e2e4"},
        {"sample_id": "g2:0", "fen": "same", "target_move": "d2d4"},
    ]

    result = evaluate_predictions(predictions, labels=labels, k=1)

    assert result["num_samples"] == 2
    assert result["top_1_accuracy"] == 1.0


def test_mock_model_returns_legal_moves():
    from chess_training_gym.board import START_FEN, legal_moves, parse_fen

    model = MockedModel(seed=3)
    moves = model.predict_top_moves(START_FEN, k=5)
    assert moves == MockedModel(seed=3).predict_top_moves(START_FEN, k=5)
    assert set(moves) <= set(legal_moves(parse_fen(START_FEN)))
