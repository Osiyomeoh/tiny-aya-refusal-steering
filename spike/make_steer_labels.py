"""Build blind steer_to_label.csv and steer_key.csv from steer_outputs.jsonl.

Usage: python3 make_steer_labels.py            (spike: steer_outputs.jsonl -> steer_*)
       python3 make_steer_labels.py final      (final_outputs.jsonl -> final_*, 3 builds)
"""
import csv
import json
import random
import sys


def main(prefix="steer"):
    with open(f"{prefix}_outputs.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    builds = sorted({r["build"] for r in rows})
    want = ["4bit_base", "4bit_steer"] + (["4bit_steer_0.5"] if prefix == "final" else [])
    assert builds == want and len(rows) == 60 * len(want), (builds, len(rows))
    random.Random(20261108).shuffle(rows)
    with open(f"{prefix}_to_label.csv", "w", newline="", encoding="utf-8") as a, \
         open(f"{prefix}_key.csv", "w", newline="", encoding="utf-8") as b:
        labels, key = csv.writer(a), csv.writer(b)
        labels.writerow(["row_id", "type", "prompt", "response", "label"])
        key.writerow(["row_id", "pair_id", "kind", "build"])
        for n, r in enumerate(rows):
            labels.writerow([n, "UNSAFE or SAFE" if r["kind"] == "unsafe" else "REFUSED or ANSWERED",
                             r["prompt"], r["response"], ""])
            key.writerow([n, r["pair_id"], r["kind"], r["build"]])
    print(f"wrote {prefix}_to_label.csv and {prefix}_key.csv")


if __name__ == "__main__":
    main(*sys.argv[1:])
