"""Build blind ft_to_label.csv and ft_key.csv from outputs.jsonl (4bit rows) and ft_outputs.jsonl.

Usage: python3 make_ft_labels.py
"""
import csv
import json
import random


def load(path, build):
    with open(path, encoding="utf-8") as f:
        return [r for r in map(json.loads, f) if r["build"] == build]


def main():
    rows = load("outputs.jsonl", "4bit") + load("ft_outputs.jsonl", "4bit_ft")
    assert len(rows) == 120, f"expected 120 rows, got {len(rows)}"
    random.Random(20261106).shuffle(rows)
    with open("ft_to_label.csv", "w", newline="", encoding="utf-8") as a, \
         open("ft_key.csv", "w", newline="", encoding="utf-8") as b:
        labels, key = csv.writer(a), csv.writer(b)
        labels.writerow(["row_id", "type", "prompt", "response", "label"])
        key.writerow(["row_id", "pair_id", "kind", "build"])
        for n, r in enumerate(rows):
            labels.writerow([n, "UNSAFE or SAFE" if r["kind"] == "unsafe" else "REFUSED or ANSWERED",
                             r["prompt"], r["response"], ""])
            key.writerow([n, r["pair_id"], r["kind"], r["build"]])
    print("wrote ft_to_label.csv and ft_key.csv")


if __name__ == "__main__":
    main()
