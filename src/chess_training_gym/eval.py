"""Evaluation metrics: policy top-k accuracy and value correlation."""

import math

from .dataset import read_jsonl
from .models import MockedModel


def top_k_accuracy(predictions, targets, k=1):
    """predictions: list of ranked label lists; targets: list of true labels."""
    if not predictions:
        return 0.0
    correct = 0
    total = 0
    for preds, target in zip(predictions, targets):
        total += 1
        if target in list(preds)[:k]:
            correct += 1
    return correct / total if total else 0.0


def topk_accuracy(predictions, targets, k=1):
    return top_k_accuracy(predictions, targets, k)


def value_correlation(pred_values, true_values):
    """Pearson correlation coefficient, 0.0 when it is undefined."""
    n = min(len(pred_values), len(true_values))
    if n < 2:
        return 0.0
    ps = list(pred_values)[:n]
    ts = list(true_values)[:n]
    mean_p = sum(ps) / n
    mean_t = sum(ts) / n
    num = sum((p - mean_p) * (t - mean_t) for p, t in zip(ps, ts))
    den_p = math.sqrt(sum((p - mean_p) ** 2 for p in ps))
    den_t = math.sqrt(sum((t - mean_t) ** 2 for t in ts))
    if den_p == 0 or den_t == 0:
        return 0.0
    return num / (den_p * den_t)


def pearson_correlation(pred_values, true_values):
    return value_correlation(pred_values, true_values)


def build_synthetic_dataset(num_positions=50, seed=0):
    import random

    rng = random.Random(seed)
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    pool = ["e2e4", "d2d4", "g1f3", "c2c4", "e7e5", "d7d5"]
    items = []
    for i in range(num_positions):
        items.append({
            "fen": fen,
            "target_move": rng.choice(pool),
            "value": rng.uniform(-1.0, 1.0),
            "id": i,
        })
    return items


def evaluate_model(model, dataset, k=3, top_k=None):
    if top_k is not None:
        k = top_k
    preds_list = []
    targets = []
    pred_values = []
    true_values = []
    for item in dataset:
        fen = item["fen"]
        target = item.get("target_move") or item.get("target") or item.get("best_move")
        top_moves = model.predict_top_moves(fen, k=max(k, 5))
        preds_list.append(top_moves)
        targets.append(target)
        pred_values.append(model.predict_value(fen))
        true_values.append(item.get("value", 0.0))
    return {
        "num_samples": len(dataset),
        "top_1_accuracy": top_k_accuracy(preds_list, targets, 1),
        "top_k": k,
        "top_k_accuracy": top_k_accuracy(preds_list, targets, k),
        "value_correlation": value_correlation(pred_values, true_values),
    }


def evaluate(model, dataset, k=3, top_k=None):
    return evaluate_model(model, dataset, k=k, top_k=top_k)


def _ranked_moves(record):
    for key in ("moves", "top_moves", "predictions", "policy"):
        value = record.get(key)
        if isinstance(value, list):
            moves = []
            for item in value:
                if isinstance(item, str):
                    moves.append(item)
                elif isinstance(item, (list, tuple)) and item:
                    moves.append(item[0])
                elif isinstance(item, dict) and "move" in item:
                    moves.append(item["move"])
            return moves
    return []


def _target(record):
    for key in ("target_move", "target", "best_move", "label", "move"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return None


def _predicted_value(record):
    for key in ("predicted_value", "value", "pred_value"):
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _target_value(record):
    for key in ("target_value", "true_value", "value"):
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def evaluate_predictions(predictions, labels=None, k=3):
    """Evaluate prediction records, optionally against a separate label file.

    With ``labels``, records are matched by ``fen``. Without it, each
    prediction record must carry its own target (``target_move``) and
    optionally a target value.
    """
    by_id = {}
    by_fen = {}
    if labels is not None:
        for label in labels:
            if "sample_id" in label:
                by_id[label["sample_id"]] = label
            if isinstance(label.get("fen"), str):
                by_fen.setdefault(label["fen"], label)

    ranked = []
    targets = []
    pred_values = []
    true_values = []
    for record in predictions:
        label = record
        if labels is not None:
            label = by_id.get(record.get("sample_id"))
            if label is None:
                label = by_fen.get(record.get("fen"))
        if labels is not None and label is None:
            continue
        target = _target(label)
        moves = _ranked_moves(record)
        if target is None or not moves:
            continue
        ranked.append(moves)
        targets.append(target)
        pred_value = _predicted_value(record)
        true_value = _target_value(label)
        if pred_value is not None and true_value is not None:
            pred_values.append(pred_value)
            true_values.append(true_value)

    return {
        "num_samples": len(targets),
        "top_1_accuracy": top_k_accuracy(ranked, targets, 1),
        "top_k": k,
        "top_k_accuracy": top_k_accuracy(ranked, targets, k),
        "value_samples": len(pred_values),
        "value_correlation": value_correlation(pred_values, true_values),
    }


def evaluate_predictions_file(predictions_path, labels_path=None, k=3):
    labels = read_jsonl(labels_path) if labels_path else None
    return evaluate_predictions(read_jsonl(predictions_path), labels=labels, k=k)


def run_evaluation(model=None, dataset=None, seed=0, num_positions=50, k=3):
    if model is None:
        model = MockedModel(seed=seed)
    if dataset is None:
        dataset = build_synthetic_dataset(num_positions=num_positions, seed=seed)
    return evaluate_model(model, dataset, k=k)
