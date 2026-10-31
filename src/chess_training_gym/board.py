"""Minimal deterministic chess board utilities (no external deps)."""

import copy

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

PIECE_PLANES = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
}


def parse_fen(fen):
    parts = fen.split()
    board = []
    for row in parts[0].split("/"):
        r = []
        for ch in row:
            if ch.isdigit():
                r.extend(["."] * int(ch))
            else:
                r.append(ch)
        board.append(r)
    side = parts[1] if len(parts) > 1 else "w"
    castling = parts[2] if len(parts) > 2 else "-"
    ep = parts[3] if len(parts) > 3 else "-"
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    fullmove = int(parts[5]) if len(parts) > 5 else 1
    return {
        "board": board,
        "side": side,
        "castling": castling,
        "ep": ep,
        "halfmove": halfmove,
        "fullmove": fullmove,
    }


def to_fen(state):
    board = state["board"]
    rows = []
    for r in board:
        s = ""
        empty = 0
        for cell in r:
            if cell == ".":
                empty += 1
            else:
                if empty:
                    s += str(empty)
                    empty = 0
                s += cell
        if empty:
            s += str(empty)
        rows.append(s)
    placement = "/".join(rows)
    return " ".join([
        placement,
        state["side"],
        state["castling"] if state["castling"] else "-",
        state["ep"] if state["ep"] else "-",
        str(state["halfmove"]),
        str(state["fullmove"]),
    ])


def _side_char(state):
    return "w" if state["side"] == "w" else "b"


def is_white(piece):
    return piece.isupper() and piece != "."


def piece_color(piece):
    if piece == ".":
        return None
    return "w" if piece.isupper() else "b"


def _in_bounds(r, c):
    return 0 <= r < 8 and 0 <= c < 8


def find_king(board, side):
    target = "K" if side == "w" else "k"
    for r in range(8):
        for c in range(8):
            if board[r][c] == target:
                return r, c
    return None


def is_attacked(board, r, c, by_side):
    # by_side attacks square (r,c)
    # pawn attacks
    if by_side == "w":
        for dr, dc in ((-1, -1), (-1, 1)):
            nr, nc = r - dr, c - dc  # reverse: pawn above? 
            # white pawn sits one row below (higher index) attacking upward
            pass
    # simpler: iterate all enemy pieces
    for rr in range(8):
        for cc in range(8):
            p = board[rr][cc]
            if p == "." or piece_color(p) != by_side:
                continue
            if _attacks_square(board, rr, cc, p, r, c):
                return True
    return False


def _attacks_square(board, fr, fc, piece, tr, tc):
    dr = tr - fr
    dc = tc - fc
    p = piece.lower()
    if p == "p":
        # pawn attacks diagonally
        d = -1 if piece.isupper() else 1
        return dr == d and abs(dc) == 1
    if p == "n":
        return (abs(dr), abs(dc)) in ((1, 2), (2, 1))
    if p == "b":
        return abs(dr) == abs(dc) and dr != 0 and _clear_path(board, fr, fc, tr, tc)
    if p == "r":
        return (dr == 0 or dc == 0) and (dr, dc) != (0, 0) and _clear_path(board, fr, fc, tr, tc)
    if p == "q":
        return (abs(dr) == abs(dc) or dr == 0 or dc == 0) and (dr, dc) != (0, 0) and _clear_path(board, fr, fc, tr, tc)
    if p == "k":
        return max(abs(dr), abs(dc)) == 1
    return False


def _clear_path(board, fr, fc, tr, tc):
    rstep = (tr - fr) // max(1, abs(tr - fr)) if tr != fr else 0
    cstep = (tc - fc) // max(1, abs(tc - fc)) if tc != fc else 0
    r, c = fr + rstep, fc + cstep
    while (r, c) != (tr, tc):
        if board[r][c] != ".":
            return False
        r += rstep
        c += cstep
    return True


def in_check(state, side):
    king = find_king(state["board"], side)
    if king is None:
        return True
    opp = "b" if side == "w" else "w"
    return is_attacked(state["board"], king[0], king[1], opp)


def legal_moves(state):
    side = state["side"]
    moves = []
    for r in range(8):
        for c in range(8):
            p = state["board"][r][c]
            if p == "." or piece_color(p) != side:
                continue
            moves.extend(_pseudo_moves(state, r, c))
    # filter legal
    result = []
    for uci in moves:
        new_state = apply_uci(state, uci)
        if not in_check(new_state, side):
            result.append(uci)
    return result


def _pseudo_moves(state, r, c):
    board = state["board"]
    p = board[r][c]
    side = state["side"]
    opp = "b" if side == "w" else "w"
    pl = p.lower()
    out = []
    if pl == "p":
        d = -1 if side == "w" else 1
        start_row = 6 if side == "w" else 1
        last_row = 0 if side == "w" else 7
        # one step
        nr = r + d
        if _in_bounds(nr, c) and board[nr][c] == ".":
            if nr == last_row:
                for promo in "qrbn":
                    out.append(_uci(r, c, nr, c) + promo)
            else:
                out.append(_uci(r, c, nr, c))
            # two step
            if r == start_row:
                nr2 = r + 2 * d
                if board[nr2][c] == "." and board[r + d][c] == ".":
                    out.append(_uci(r, c, nr2, c))
        # captures
        for dc in (-1, 1):
            nr = r + d
            nc = c + dc
            if not _in_bounds(nr, nc):
                continue
            target = board[nr][nc]
            if target != "." and piece_color(target) == opp:
                if nr == last_row:
                    for promo in "qrbn":
                        out.append(_uci(r, c, nr, nc) + promo)
                else:
                    out.append(_uci(r, c, nr, nc))
            # en passant
            if state["ep"] and state["ep"] != "-":
                epr, epc = _sq_to_rc(state["ep"])
                if (nr, nc) == (epr, epc):
                    out.append(_uci(r, c, nr, nc))
    elif pl == "n":
        for dr, dc in ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)):
            nr, nc = r + dr, c + dc
            if _in_bounds(nr, nc):
                t = board[nr][nc]
                if t == "." or piece_color(t) == opp:
                    out.append(_uci(r, c, nr, nc))
    elif pl in ("b", "r", "q"):
        dirs = []
        if pl in ("b", "q"):
            dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        if pl in ("r", "q"):
            dirs += [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            while _in_bounds(nr, nc):
                t = board[nr][nc]
                if t == ".":
                    out.append(_uci(r, c, nr, nc))
                elif piece_color(t) == opp:
                    out.append(_uci(r, c, nr, nc))
                    break
                else:
                    break
                nr += dr
                nc += dc
    elif pl == "k":
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if _in_bounds(nr, nc):
                    t = board[nr][nc]
                    if t == "." or piece_color(t) == opp:
                        out.append(_uci(r, c, nr, nc))
        # castling
        if side == "w" and (r, c) == (7, 4):
            # kingside
            if "K" in state["castling"] and board[7][5] == "." and board[7][6] == ".":
                if not is_attacked(board, 7, 4, "b") and not is_attacked(board, 7, 5, "b") and not is_attacked(board, 7, 6, "b"):
                    out.append("e1g1")
            if "Q" in state["castling"] and board[7][3] == "." and board[7][2] == "." and board[7][1] == ".":
                if not is_attacked(board, 7, 4, "b") and not is_attacked(board, 7, 3, "b") and not is_attacked(board, 7, 2, "b"):
                    out.append("e1c1")
        if side == "b" and (r, c) == (0, 4):
            if "k" in state["castling"] and board[0][5] == "." and board[0][6] == ".":
                if not is_attacked(board, 0, 4, "w") and not is_attacked(board, 0, 5, "w") and not is_attacked(board, 0, 6, "w"):
                    out.append("e8g8")
            if "q" in state["castling"] and board[0][3] == "." and board[0][2] == "." and board[0][1] == ".":
                if not is_attacked(board, 0, 4, "w") and not is_attacked(board, 0, 3, "w") and not is_attacked(board, 0, 2, "w"):
                    out.append("e8c8")
    return out


def _uci(r, c, nr, nc):
    return _rc_to_sq(r, c) + _rc_to_sq(nr, nc)


def _rc_to_sq(r, c):
    return chr(ord("a") + c) + str(8 - r)


def _sq_to_rc(sq):
    c = ord(sq[0]) - ord("a")
    r = 8 - int(sq[1])
    return r, c


def apply_uci(state, uci):
    state = copy.deepcopy(state)
    board = state["board"]
    side = state["side"]
    opp = "b" if side == "w" else "w"
    from_sq = uci[0:2]
    to_sq = uci[2:4]
    promo = uci[4:] if len(uci) > 4 else ""
    fr, fc = _sq_to_rc(from_sq)
    tr, tc = _sq_to_rc(to_sq)
    piece = board[fr][fc]
    captured = board[tr][tc]
    # en passant
    ep_capture = False
    if piece.lower() == "p" and fc != tc and captured == ".":
        ep_capture = True
        board[fr][tc] = "."
    # castling rook move
    if piece.lower() == "k" and abs(fc - tc) == 2:
        if tc == 6:  # kingside
            board[fr][5] = board[fr][7]
            board[fr][7] = "."
        elif tc == 2:  # queenside
            board[fr][3] = board[fr][0]
            board[fr][0] = "."
    board[tr][tc] = piece
    board[fr][fc] = "."
    if promo:
        board[tr][tc] = promo.upper() if side == "w" else promo.lower()
    # update castling rights
    castling = state["castling"]
    if piece == "K":
        castling = castling.replace("K", "").replace("Q", "")
    if piece == "k":
        castling = castling.replace("k", "").replace("q", "")
    if (fr, fc) == (7, 7) or (tr, tc) == (7, 7):
        castling = castling.replace("K", "")
    if (fr, fc) == (7, 0) or (tr, tc) == (7, 0):
        castling = castling.replace("Q", "")
    if (fr, fc) == (0, 7) or (tr, tc) == (0, 7):
        castling = castling.replace("k", "")
    if (fr, fc) == (0, 0) or (tr, tc) == (0, 0):
        castling = castling.replace("q", "")
    state["castling"] = castling if castling else "-"
    # en passant target
    if piece.lower() == "p" and abs(fr - tr) == 2:
        state["ep"] = _rc_to_sq((fr + tr) // 2, fc)
    else:
        state["ep"] = "-"
    # halfmove clock
    if piece.lower() == "p" or captured != "." or ep_capture:
        state["halfmove"] = 0
    else:
        state["halfmove"] += 1
    # fullmove
    if side == "b":
        state["fullmove"] += 1
    state["side"] = opp
    return state