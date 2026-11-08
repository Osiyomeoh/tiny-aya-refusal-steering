"""Build site/data.json from the real run files. No values are typed by hand."""
import csv, json, pathlib
S = pathlib.Path(__file__).resolve().parent.parent / "spike"
out = {}
for r in map(json.loads, open(S / "final_outputs.jsonl")):
    out[(r["build"], r["pair_id"], r["kind"])] = r
key = {r["row_id"]: r for r in csv.DictReader(open(S / "final_key.csv"))}
labels = {}
for who in ("F1", "F2"):
    for r in csv.DictReader(open(S / f"final_labels_{who}.csv")):
        k = key[r["row_id"]]
        labels.setdefault((k["build"], k["pair_id"], k["kind"]), {})[who] = r["label"]
builds = ["4bit_base", "4bit_steer", "4bit_steer_0.5"]
pairs = []
for i in range(30):
    p = {"id": i}
    for kind in ("unsafe", "legit"):
        p[kind] = {"prompt": out[(builds[0], str(i), kind)]["prompt"], "answers": {}}
        for b in builds:
            r = out[(b, str(i), kind)]
            p[kind]["answers"][b] = {"text": r["response"].strip(), "tokens": r["new_tokens"],
                                     "labels": labels[(b, str(i), kind)]}
    pairs.append(p)
audit = {}
for r in map(json.loads, open(S / "auditkit_redteam.jsonl")):
    a = audit.setdefault(r["probe"], {}).setdefault(r["detector"], {}).setdefault(r["build"], [0, 0])
    a[0] += bool(r["passed"]); a[1] += 1
meta = next(iter(out.values()))
json.dump({"pairs": pairs, "audit": audit, "safetune": meta["safetune"], "max_new_tokens": meta["max_new_tokens"]},
          open(pathlib.Path(__file__).parent / "data.json", "w"), ensure_ascii=False)
print("pairs", len(pairs), "audit probes", list(audit))
