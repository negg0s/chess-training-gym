from chess_training_gym.board import (
    START_FEN,
    apply_uci,
    in_check,
    legal_moves,
    parse_fen,
    to_fen,
)


def perft(state, depth):
    if depth == 0:
        return 1
    return sum(perft(apply_uci(state, move), depth - 1) for move in legal_moves(state))


def test_start_position_move_count():
    state = parse_fen(START_FEN)
    moves = legal_moves(state)
    assert len(moves) == 20
    assert "e2e4" in moves
    assert "g1f3" in moves


def test_perft_depth_three():
    assert perft(parse_fen(START_FEN), 3) == 8902


def test_kiwipete_depth_one():
    fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    assert len(legal_moves(parse_fen(fen))) == 48


def test_perft_promotion_position():
    fen = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"
    assert perft(parse_fen(fen), 2) == 1486


def test_castling_moves_available():
    state = parse_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    moves = legal_moves(state)
    assert "e1g1" in moves
    assert "e1c1" in moves


def test_castling_blocked_by_attack():
    state = parse_fen("r3k2r/8/8/8/8/8/5q2/R3K2R w KQkq - 0 1")
    moves = legal_moves(state)
    assert "e1g1" not in moves


def test_castling_rook_moves():
    state = apply_uci(parse_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"), "e1g1")
    assert state["board"][7][6] == "K"
    assert state["board"][7][5] == "R"
    assert state["board"][7][7] == "."
    assert state["side"] == "b"


def test_en_passant():
    state = parse_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    assert "e5d6" in legal_moves(state)
    after = apply_uci(state, "e5d6")
    assert after["board"][3][3] == "."
    assert after["board"][2][3] == "P"


def test_promotion():
    state = parse_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    moves = legal_moves(state)
    assert "a7a8q" in moves
    after = apply_uci(state, "a7a8q")
    assert after["board"][0][0] == "Q"


def test_fools_mate_is_checkmate():
    state = parse_fen(START_FEN)
    for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
        state = apply_uci(state, move)
    assert in_check(state, "w")
    assert legal_moves(state) == []


def test_stalemate_has_no_moves():
    state = parse_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert not in_check(state, "b")
    assert legal_moves(state) == []


def test_fen_roundtrip():
    fen = "r3k2r/pp1n1ppp/2p1pn2/3p4/3P4/2N1PN2/PP3PPP/R3K2R b KQkq - 3 9"
    assert to_fen(parse_fen(fen)) == fen


def test_invalid_fen_rejected():
    for fen in (
        "",
        "8/8/8",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x - - 0 1",
        "8/8/8/8/8/8/8/8 w - z9 0 1",
        "8/8/8/8/8/8/8/8 w - - -1 1",
        "8/8/8/8/8/8/8/8 w - - 0 0",
    ):
        try:
            parse_fen(fen)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid FEN: {fen!r}")
