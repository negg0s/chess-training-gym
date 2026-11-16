"""PGN (Portable Game Notation) reader and parser utilities."""

from .board import START_FEN, _sq_to_rc, apply_uci, legal_moves, parse_fen, to_fen


def resolve_san_move(state, token):
    """Resolve a SAN or UCI move token into a legal UCI move string.

    Raises ValueError if the move is malformed or illegal in the given position.
    """
    if not isinstance(token, str) or not token:
        raise ValueError("move token must be a non-empty string")

    # Strip check, checkmate, annotations
    clean = token.rstrip("+#?!")
    if not clean:
        raise ValueError(f"invalid move token: {token!r}")

    legal = legal_moves(state)

    # Direct UCI match
    if clean in legal:
        return clean

    # Castling notation
    if clean in ("O-O", "0-0"):
        uci = "e1g1" if state["side"] == "w" else "e8g8"
        if uci in legal:
            return uci
        raise ValueError(f"illegal castling {token!r} in position {to_fen(state)}")

    if clean in ("O-O-O", "0-0-0"):
        uci = "e1c1" if state["side"] == "w" else "e8c8"
        if uci in legal:
            return uci
        raise ValueError(f"illegal castling {token!r} in position {to_fen(state)}")

    # Promotion detection
    promo = ""
    base = clean
    if "=" in clean:
        parts = clean.split("=")
        base = parts[0]
        promo = parts[1].lower() if len(parts) > 1 else ""
    elif len(clean) >= 3 and clean[-1].upper() in "QRBN" and clean[-2].isdigit():
        promo = clean[-1].lower()
        base = clean[:-1]

    if len(base) < 2:
        raise ValueError(f"invalid move token: {token!r}")

    dest = base[-2:]
    if not (dest[0] in "abcdefgh" and dest[1] in "12345678"):
        raise ValueError(f"invalid destination square in {token!r}")

    prefix = base[:-2].replace("x", "")
    if not prefix:
        piece_type = "P"
        disambig = ""
    elif prefix[0] in "NBRQK":
        piece_type = prefix[0]
        disambig = prefix[1:]
    else:
        # Pawn capture with file disambiguation
        piece_type = "P"
        disambig = prefix

    matches = []
    board = state["board"]
    for uci in legal:
        if uci[2:4] != dest:
            continue
        uci_promo = uci[4:].lower() if len(uci) > 4 else ""
        if promo and uci_promo != promo:
            continue
        fr, fc = _sq_to_rc(uci[0:2])
        p = board[fr][fc]
        if p.upper() != piece_type:
            continue
        from_sq = uci[0:2]
        if disambig:
            if len(disambig) == 1:
                if disambig in "abcdefgh" and from_sq[0] != disambig:
                    continue
                if disambig in "12345678" and from_sq[1] != disambig:
                    continue
            elif len(disambig) == 2 and from_sq != disambig:
                continue
        matches.append(uci)

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"illegal move {token!r} in position {to_fen(state)}")
    raise ValueError(f"ambiguous move {token!r} in position {to_fen(state)}: candidates={matches}")


def _tokenize_pgn(text):
    """Tokenize PGN text into tags, movetext tokens, and comments.

    Validates that comments, string literals, tags, and variations are closed properly,
    raising ValueError on unexpected EOF.
    """
    tokens = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue

        # Comments with semicolon run to end of line
        if ch == ";":
            start = i
            while i < n and text[i] not in "\r\n":
                i += 1
            continue

        # Comments enclosed in curly braces
        if ch == "{":
            start = i
            i += 1
            found_end = False
            while i < n:
                if text[i] == "}":
                    found_end = True
                    i += 1
                    break
                i += 1
            if not found_end:
                raise ValueError("unexpected EOF: unclosed comment in PGN")
            continue

        # Variations in parentheses
        if ch == "(":
            depth = 1
            i += 1
            while i < n and depth > 0:
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                i += 1
            if depth > 0:
                raise ValueError("unexpected EOF: unclosed variation in PGN")
            continue

        # Tag pair [Key "Value"]
        if ch == "[":
            tag_start = i
            i += 1
            while i < n and text[i].isspace():
                i += 1
            key_start = i
            while i < n and (text[i].isalnum() or text[i] == "_"):
                i += 1
            key = text[key_start:i]

            # Find quote
            while i < n and text[i] != '"' and text[i] != "]":
                i += 1
            if i >= n:
                raise ValueError("unexpected EOF: unclosed tag bracket in PGN")
            if text[i] != '"':
                raise ValueError("malformed PGN tag: missing quote")

            i += 1  # skip opening quote
            val_start = i
            escaped = False
            val_chars = []
            quote_closed = False
            while i < n:
                curr = text[i]
                if escaped:
                    val_chars.append(curr)
                    escaped = False
                elif curr == "\\":
                    escaped = True
                elif curr == '"':
                    quote_closed = True
                    i += 1
                    break
                else:
                    val_chars.append(curr)
                i += 1

            if not quote_closed:
                raise ValueError("unexpected EOF in PGN header string")

            while i < n and text[i] != "]":
                if not text[i].isspace():
                    raise ValueError(f"unexpected character in tag: {text[i]!r}")
                i += 1
            if i >= n:
                raise ValueError("unexpected EOF: unclosed tag bracket in PGN")
            i += 1  # skip closing ]

            tokens.append(("TAG", key, "".join(val_chars)))
            continue

        # Move token, move number, or game result
        tok_start = i
        while i < n and not text[i].isspace() and text[i] not in "[]{};()":
            i += 1
        raw_token = text[tok_start:i]

        # Ignore move numbers like 1. or 12...
        if raw_token.endswith(".") or raw_token.startswith("$"):
            continue
        # Also handle cases like 1.e4 where move number is attached
        if "." in raw_token:
            parts = raw_token.split(".")
            move_candidate = parts[-1]
            if move_candidate:
                tokens.append(("MOVE", move_candidate))
        else:
            tokens.append(("MOVE", raw_token))

    return tokens


def parse_pgn(text):
    """Parse PGN text into a list of game records.

    Each game record is a dictionary containing:
      - headers: dict of tag key/value pairs
      - moves: list of moves in UCI notation
      - positions: list of FEN strings from initial to final
      - result: game result string ('1-0', '0-1', '1/2-1/2', or '*')
      - num_plies: total number of moves
    """
    if not isinstance(text, str):
        raise ValueError("PGN input must be a string")
    if not text.strip():
        return []

    tokens = _tokenize_pgn(text)
    games = []
    current_headers = {}
    current_moves = []
    current_fen = None
    state = None

    def finalize_game(result="*"):
        nonlocal current_headers, current_moves, current_fen, state
        if not current_headers and not current_moves:
            return
        res = current_headers.get("Result", result)
        if res == "*" and result != "*":
            res = result
        games.append({
            "headers": dict(current_headers),
            "moves": list(current_moves),
            "positions": list(positions),
            "result": res,
            "num_plies": len(current_moves),
        })
        current_headers = {}
        current_moves = []
        current_fen = None
        state = None

    positions = []
    for item in tokens:
        kind = item[0]
        if kind == "TAG":
            key, val = item[1], item[2]
            if current_moves:
                finalize_game()
                positions = []
            current_headers[key] = val
            if key == "FEN":
                current_fen = val
        elif kind == "MOVE":
            token = item[1]
            if token in ("1-0", "0-1", "1/2-1/2", "*"):
                finalize_game(result=token)
                positions = []
                continue

            if state is None:
                start = current_headers.get("FEN", START_FEN)
                state = parse_fen(start)
                positions = [to_fen(state)]

            uci = resolve_san_move(state, token)
            state = apply_uci(state, uci)
            current_moves.append(uci)
            positions.append(to_fen(state))

    if current_headers or current_moves:
        finalize_game()

    return games


def read_pgn(path):
    """Read a PGN file and return parsed game records."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return parse_pgn(content)


pgn_to_games = parse_pgn
