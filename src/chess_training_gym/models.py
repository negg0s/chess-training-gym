"""Mocked models for evaluation utilities."""

import hashlib
import random

from .board import legal_moves, parse_fen


def _stable_seed(*parts):
    text = "|".join(str(part) for part in parts)
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:16], 16)


class MockedModel:
    """A deterministic model used to test the evaluation pipeline.

    It ranks legal moves with pseudo-random scores. It does not estimate
    chess strength.
    """

    def __init__(self, seed=0, perfect=False, num_candidates=8):
        self.seed = seed
        self.perfect = perfect
        self.num_candidates = num_candidates

    def _moves_pool(self, fen):
        try:
            moves = sorted(legal_moves(parse_fen(fen)))
        except ValueError:
            moves = []
        if not moves:
            rng = random.Random(_stable_seed(fen, self.seed))
            files = "abcdefgh"
            ranks = "12345678"
            for _ in range(self.num_candidates):
                moves.append(rng.choice(files) + rng.choice(ranks)
                             + rng.choice(files) + rng.choice(ranks))
        rng = random.Random(_stable_seed(fen, self.seed, "pool"))
        rng.shuffle(moves)
        return moves[:self.num_candidates]

    def predict(self, fen):
        return {
            "policy": self.predict_policy(fen),
            "value": self.predict_value(fen),
        }

    def predict_policy(self, fen):
        rng = random.Random(_stable_seed(fen, self.seed, "policy"))
        scored = [(move, rng.random()) for move in self._moves_pool(fen)]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def predict_top_moves(self, fen, k=5):
        return [move for move, _ in self.predict_policy(fen)[:k]]

    def predict_topk(self, fen, k=5):
        return self.predict_top_moves(fen, k)

    def top_moves(self, fen, k=5):
        return self.predict_top_moves(fen, k)

    def predict_value(self, fen):
        rng = random.Random(_stable_seed(fen, self.seed, "value"))
        return rng.uniform(-1.0, 1.0)

    def predict_for(self, fen, target_move=None, true_value=None):
        if self.perfect and target_move is not None:
            return {
                "policy": [(target_move, 1.0)],
                "value": true_value if true_value is not None else 0.0,
            }
        return self.predict(fen)
