"""Report generation from evaluation results."""

import json


def generate_report(results, format="markdown"):
    if format == "json":
        return json.dumps(results, indent=2)
    lines = ["# Chess Training Gym Report", ""]
    for key, val in results.items():
        lines.append(f"- **{key}**: {val}")
    lines.append("")
    return "\n".join(lines)


def make_report(results, format="markdown"):
    return generate_report(results, format=format)


def write_report(results, path, format="markdown"):
    content = generate_report(results, format=format)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def read_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)