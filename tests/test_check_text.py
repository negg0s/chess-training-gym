from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import check_text


def test_clean_file(tmp_path):
    path = tmp_path / "clean.txt"
    path.write_text("plain words with a hyphen\n", encoding="utf-8")
    assert check_text.scan(path) == []


def test_detects_dashes(tmp_path):
    path = tmp_path / "dash.txt"
    path.write_text("em \u2014 dash and en \u2013 dash\n", encoding="utf-8")
    assert len(check_text.scan(path)) == 2


def test_detects_banned_phrase(tmp_path):
    path = tmp_path / "phrase.txt"
    path.write_text("we " + "lever" + "age the data\n", encoding="utf-8")
    assert check_text.scan(path)
