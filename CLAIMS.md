# CLAIMS (each tied to a run or a source)

| # | Claim | Evidence | Status |
|---|---|---|---|
| 1 | Steering cut unsafe compliance on 30 paired dangerous prompts: 11/30→4/30 (A1), 12/30→5/30 (A2) | spike/steer_outputs.jsonl, steer_labels_A1/A2.csv, scoring.py | run, 2026-11-07 |
| 2 | Net gain 7 (8 better, 1 worse) for both labellers; exact McNemar p=0.039 | same | run |
| 3 | No over-refusal: 0/30 legit questions refused, default or steered | same | run; weak (no base refusals) |
| 4 | Labellers disagree on 4/120 rows (30, 50, 56, 61) | label CSVs | run |
| 5 | The default build reproduces the earlier 4bit run exactly (60/60) | outputs.jsonl vs steer_outputs.jsonl | run |
| 6 | Direction from layer 18 (middle layer, no sweep), strength 0.25 | steer_cell.py, STEER-RULE.md | config |
| 7 | Prior labeller agreement: H–A1 κ 0.724, H–A2 κ 0.795, A1–A2 κ 0.93 | agree_*.csv | run (earlier set) |

Limits: 108/120 answers hit the 200-token cap. There are only 30 pairs. One layer and one strength. About 6 legit answers misread the question.
Not claimed: multilingual effect, generalisation beyond electrical safety, robustness to jailbreaks.

## Final run (512 tokens, 2026-11-07/08, labelled F1/F2 blind)
| # | Claim | Evidence | Status |
|---|---|---|---|
| 8 | Fixed config (layer 18, strength 0.25): unsafe compliance 11/30→5/30, 6 better / 0 worse, net 6, exact McNemar p=0.031, for both labellers | spike/final_outputs.jsonl, final_labels_F1/F2.csv, steer_score.py | run; steer rule PROCEED |
| 9 | Over-refusal at 0.25: 0/30, both default and steered | same | run |
| 10 | Exploratory strength 0.5: 11/30→3/30 (F1) or 4/30 (F2), net 8/7, p=0.021/0.039; 1/30 legit refused | same | exploratory; not used for the decision |
| 11 | F1–F2 disagree on 3/180 rows (15, 58, 131) | label CSVs | run |
| 12 | 23/180 answers hit the 512 cap (spike: 108/120 hit 200) | final_outputs.jsonl | run |
| 13 | AuditKIT 1.1.6 red-team, each probe with its mapped detectors: base and steered 0.25 identical on every probe/detector (injection 5/8, jailbreak 8/8, encoding 4/5, over-refusal 5/5); outputs differ on 39/47 checks, no pass/fail flipped. Keyword/regex detectors, 5–8 prompts per probe | spike/auditkit_redteam.jsonl | run; null result |
