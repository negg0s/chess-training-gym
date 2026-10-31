from chess_training_gym.eval import top_k_accuracy, value_correlation, evaluate_model, build_synthetic_dataset
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