from chess_training_gym.encode import fen_to_tensor, encode_position, batch_encode, read_positions

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


def test_tensor_planes():
    tensor = fen_to_tensor(FEN)
    if hasattr(tensor, "shape"):
        assert tensor[0, 6, 0] == 1.0
        assert tensor[12, 0, 0] == 1.0
    else:
        assert tensor[0][6][0] == 1.0
        assert tensor[12][0][0] == 1.0


def test_read_positions_fen_file(tmp_path):
    path = tmp_path / "positions.txt"
    path.write_text(FEN + "\n" + FEN + "\n", encoding="utf-8")
    assert read_positions(path) == [FEN, FEN]


def test_read_positions_jsonl(tmp_path):
    path = tmp_path / "games.jsonl"
    path.write_text(
        '{"type": "metadata", "seed": 1}\n'
        + '{"positions": ["%s"]}\n' % FEN
        + '{"fen": "%s"}\n' % FEN,
        encoding="utf-8",
    )
    assert read_positions(path) == [FEN, FEN]


def test_read_positions_json_array(tmp_path):
    path = tmp_path / "positions.json"
    path.write_text('["%s", "%s"]' % (FEN, FEN), encoding="utf-8")
    assert read_positions(path) == [FEN, FEN]
