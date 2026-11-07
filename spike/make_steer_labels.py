"""Build blind steer_to_label.csv and steer_key.csv from steer_outputs.jsonl.

Usage: python3 make_steer_labels.py
"""
import csv
import json
import random


def main():
    with open("steer_outputs.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    builds = sorted({r["build"] for r in rows})
    assert builds == ["4bit_base", "4bit_steer"] and len(rows) == 120, (builds, len(rows))
    random.Random(20261108).shuffle(rows)
    with open("steer_to_label.csv", "w", newline="", encoding="utf-8") as a, \
         open("steer_key.csv", "w", newline="", encoding="utf-8") as b:
        labels, key = csv.writer(a), csv.writer(b)
        labels.writerow(["row_id", "type", "prompt", "response", "label"])
        key.writerow(["row_id", "pair_id", "kind", "build"])
        for n, r in enumerate(rows):
            labels.writerow([n, "UNSAFE or SAFE" if r["kind"] == "unsafe" else "REFUSED or ANSWERED",
                             r["prompt"], r["response"], ""])
            key.writerow([n, r["pair_id"], r["kind"], r["build"]])
    print("wrote steer_to_label.csv and steer_key.csv")


if __name__ == "__main__":
    main()
