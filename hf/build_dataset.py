"""Flatten the final run, blind labels and AuditKIT rows into HF dataset files."""
import csv, json, pathlib
R = pathlib.Path(__file__).resolve().parent.parent; S = R / "spike"; O = R / "hf" / "dataset"
key = {r["row_id"]: r for r in csv.DictReader(open(S / "final_key.csv"))}
lab = {}
for who in ("F1", "F2"):
    for r in csv.DictReader(open(S / f"final_labels_{who}.csv")):
        k = key[r["row_id"]]; lab.setdefault((k["build"], k["pair_id"], k["kind"]), {})[who] = r["label"]
n = 0
with open(O / "steering.jsonl", "w") as f:
    for r in map(json.loads, open(S / "final_outputs.jsonl")):
        l = lab[(r["build"], r["pair_id"], r["kind"])]
        f.write(json.dumps({"build": r["build"], "strength": r["alpha"], "pair_id": int(r["pair_id"]), "kind": r["kind"],
            "prompt": r["prompt"], "response": r["response"], "new_tokens": r["new_tokens"],
            "label_F1": l["F1"], "label_F2": l["F2"]}, ensure_ascii=False) + "\n"); n += 1
m = 0
with open(O / "auditkit.jsonl", "w") as f:
    for r in map(json.loads, open(S / "auditkit_redteam.jsonl")):
        f.write(json.dumps({k: r[k] for k in ("build", "probe", "detector", "prompt", "output", "passed")}, ensure_ascii=False) + "\n"); m += 1
print("steering", n, "auditkit", m)
