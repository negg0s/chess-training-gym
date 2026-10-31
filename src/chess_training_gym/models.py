"""Mocked models for evaluation utilities."""

import random


class MockedModel:
    """A deterministic mocked model.

    Parameters
    ----------
    seed: int
    perfect: bool
        If True, predict the provided target when possible.
    num_candidates: int
    """

    def __init__(self, seed=0, perfect=False, num_candidates=8):
        self.seed = seed
        self.perfect = perfect
        self.num_candidates = num_candidates
        self._rng = random.Random(seed)

    def _moves_pool(self, fen):
        # deterministic pseudo candidate moves
        files = "abcdefgh"
        ranks = "12345678"
        rng = random.Random(hash(fen) ^ self.seed)
        out = []
        for _ in range(self.num_candidates):
            a = rng.choice(files) + rng.choice(ranks)
            b = rng.choice(files) + rng.choice(ranks)
            out.append(a + b)
        return out

    def predict(self, fen):
        return {
            "policy": self.predict_policy(fen),
            "value": self.predict_value(fen),
        }

    def predict_policy(self, fen):
        pool = self._moves_pool(fen)
        rng = random.Random(hash(fen) ^ self.seed ^ 1)
        # random scores
        scored = [(m, rng.random()) for m in pool]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def predict_top_moves(self, fen, k=5):
        return [m for m, _ in self.predict_policy(fen)[:k]]

    def predict_topk(self, fen, k=5):
        return self.predict_top_moves(fen, k)

    def top_moves(self, fen, k=5):
        return self.predict_top_moves(fen, k)

    def predict_value(self, fen):
        rng = random.Random(hash(fen) ^ self.seed ^ 2)
        return rng.uniform(-1.0, 1.0)

    # perfect-mode helpers
    def predict_for(self, fen, target_move=None, true_value=None):
        if self.perfect and target_move is not None:
            return {"policy": [(target_move, 1.0)], "value": true_value if true_value is not None else 0.0}
        return self.predict(fen)