import pytest

from chess_training_gym.pgn import parse_pgn, read_pgn, resolve_san_move
from chess_training_gym.board import START_FEN, parse_fen


def test_empty_pgn():
    assert parse_pgn("") == []
    assert parse_pgn("   \n\t   \n") == []


def test_corrupted_pgn_unexpected_eof_header_quote():
    pgn = '[Event "Unclosed Header'
    with pytest.raises(ValueError, match="unexpected EOF in PGN header string"):
        parse_pgn(pgn)


def test_corrupted_pgn_unexpected_eof_header_bracket():
    pgn = '[Event "Missing Bracket"'
    with pytest.raises(ValueError, match="unexpected EOF: unclosed tag bracket in PGN"):
        parse_pgn(pgn)


def test_corrupted_pgn_unexpected_eof_comment():
    pgn = "1. e4 {Unclosed comment without curly brace"
    with pytest.raises(ValueError, match="unexpected EOF: unclosed comment in PGN"):
        parse_pgn(pgn)


def test_corrupted_pgn_unexpected_eof_variation():
    pgn = "1. e4 (1. d4 d5"
    with pytest.raises(ValueError, match="unexpected EOF: unclosed variation in PGN"):
        parse_pgn(pgn)


def test_invalid_move_syntax():
    pgn = "1. invalid_move"
    with pytest.raises(ValueError, match="invalid"):
        parse_pgn(pgn)


def test_illegal_move_for_turn():
    # White cannot play e5 on move 1
    pgn = "1. e5"
    with pytest.raises(ValueError, match="illegal move"):
        parse_pgn(pgn)


def test_illegal_king_positions_in_pgn_fen():
    # No white king
    pgn_no_king = '[FEN "8/8/8/8/8/8/8/4k3 w - - 0 1"]\n1. e4'
    with pytest.raises(ValueError, match="white king"):
        parse_pgn(pgn_no_king)

    # Two white kings
    pgn_two_kings = '[FEN "4K3/8/8/8/8/8/8/4k2K w - - 0 1"]\n1. e4'
    with pytest.raises(ValueError, match="white king"):
        parse_pgn(pgn_two_kings)

    # Adjacent kings
    pgn_adj_kings = '[FEN "4kK2/8/8/8/8/8/8/8 w - - 0 1"]\n1. e4'
    with pytest.raises(ValueError, match="kings cannot be adjacent"):
        parse_pgn(pgn_adj_kings)

    # Side not to move is in check
    # Side is white, but black king is in check
    pgn_opp_in_check = '[FEN "4k3/4R3/8/8/8/8/8/4K3 w - - 0 1"]\n1. Re8+'
    with pytest.raises(ValueError, match="side not to move cannot be in check"):
        parse_pgn(pgn_opp_in_check)


def test_valid_pgn_fools_mate():
    pgn = """
    [Event "World Championship"]
    [Site "London"]
    [Date "2026.01.01"]
    [White "Alice"]
    [Black "Bob"]
    [Result "0-1"]

    1. f3 e5 2. g4 Qh4# 0-1
    """
    games = parse_pgn(pgn)
    assert len(games) == 1
    g = games[0]
    assert g["headers"]["White"] == "Alice"
    assert g["headers"]["Result"] == "0-1"
    assert g["result"] == "0-1"
    assert g["moves"] == ["f2f3", "e7e5", "g2g4", "d8h4"]
    assert g["num_plies"] == 4
    assert len(g["positions"]) == 5


def test_valid_pgn_castling_and_disambiguation():
    pgn = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 *"
    games = parse_pgn(pgn)
    assert len(games) == 1
    g = games[0]
    assert g["moves"][:2] == ["e2e4", "e7e5"]
    assert "e1g1" in g["moves"]  # O-O for white


def test_read_pgn_file(tmp_path):
    file_path = tmp_path / "sample.pgn"
    file_path.write_text("1. e4 e5 2. Nf3 Nc6 1/2-1/2\n", encoding="utf-8")
    games = read_pgn(file_path)
    assert len(games) == 1
    assert games[0]["result"] == "1/2-1/2"
    assert games[0]["num_plies"] == 4
