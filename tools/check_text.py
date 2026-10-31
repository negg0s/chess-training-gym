#!/usr/bin/env python3
"""Check repository text for em/en dashes and banned filler phrases."""

import argparse
import subprocess
import sys
from pathlib import Path

DASH_CHARS = {
    "\u2014": "em dash",
    "\u2013": "en dash",
}

# Split so this file does not trip its own check.
BANNED = [
    "compre" + "hensive",
    "sea" + "mless",
    "rob" + "ust",
    "lever" + "age",
    "del" + "ve",
    "cutting" + "-edge",
    "state" + "-of-the-art",
    "game" + "-changer",
    "un" + "lock",
    "elev" + "ate",
    "effort" + "less",
    "stream" + "line",
    "har" + "ness",
    "empow" + "er",
    "revol" + "utionize",
    "in to" + "day's",
    "it's imp" + "ortant to note",
    "further" + "more",
    "more" + "over",
]

SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz",
    ".whl", ".pyc", ".so", ".dll", ".exe", ".woff", ".woff2",
}


def repo_files():
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            stderr=subprocess.DEVNULL,
        )
        return [Path(p) for p in out.decode("utf-8").split("\0") if p]
    except (OSError, subprocess.CalledProcessError):
        return [p for p in Path(".").rglob("*") if p.is_file()]


def scan(path):
    if path.suffix.lower() in SKIP_SUFFIXES:
        return []
    try:
        data = path.read_bytes()
    except OSError as exc:
        return [(0, 0, f"cannot read file: {exc}")]
    if b"\0" in data:
        return []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return []
    problems = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for char, name in DASH_CHARS.items():
            col = line.find(char)
            if col != -1:
                problems.append((lineno, col + 1, f"{name} is not allowed"))
        lower = line.lower()
        for phrase in BANNED:
            col = lower.find(phrase)
            if col != -1:
                problems.append((lineno, col + 1, f"banned phrase: {phrase!r}"))
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Files to check (default: repo files)")
    args = parser.parse_args(argv)
    paths = [Path(p) for p in args.paths] if args.paths else repo_files()
    failures = 0
    for path in paths:
        for lineno, col, message in scan(path):
            print(f"{path}:{lineno}:{col}: {message}")
            failures += 1
    if failures:
        print(f"{failures} problem(s) found", file=sys.stderr)
        return 1
    print(f"checked {len(paths)} file(s): clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
