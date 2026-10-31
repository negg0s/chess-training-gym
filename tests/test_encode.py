from chess_training_gym.encode import fen_to_tensor, encode_position, batch_encode

FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_shape():
    t = fen_to_tensor(FEN)
    try:
        shape = t.shape
    except AttributeError:
        shape = (len(t), len(t[0]), len(t[0][0]))
    assert shape == (13, 8, 8)


def test_encode_position():
    ep = encode_position(FEN)
    assert ep["fen"] == FEN
    assert ep["shape"] == (13, 8, 8)


def test_batch():
    ts = batch_encode([FEN, FEN])
    assert len(ts) == 2