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
