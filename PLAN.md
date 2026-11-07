# PLAN: inference-time safety steering on Tiny Aya Global

Track 2b, problem 8. Deadline: end of Nov 14 2026 (AoE).

## What we build
1. **Pipeline** (`steer/`): load Tiny Aya Global in NF4. Extract a refusal direction with SafeTune and steer it at inference time. Run the paired unsafe/legit prompt set, then score with blind labels and exact McNemar.
2. **Audit** (AuditKIT, the second track repo): run the default and steered builds through AuditKIT for a second before/after metric. [FILL: which AuditKIT check fits; check its README.]
3. **Demo**: a static page with a side-by-side browser of the real 120 outputs, labels and stats. Sample mode = bundled real outputs, no GPU. Check it at 390px and 1280px with 0 console errors and 0 overflow.
4. **HF**: a public collection with the outputs dataset and model card notes (Sam publishes).
5. **Write-up**: method, results, limits.

## Order (small commits)
1. Repo hygiene: .gitignore, README, move the spike results in as evidence.
2. `steer/score.py` with the steer rule as code, plus tests (fix the old `decide()`).
3. Layer/strength sweep on Colab (extend the evidence; the rule stays fixed).
4. Raise max_new_tokens above 200 for the final run, to fix truncation.
5. AuditKIT run.
6. Demo page plus a deploy (Sam does the hosting account).
7. Write-up, HF collection, submission (Sam submits).

## Stop rules
- If the final run with longer outputs falls below the steer rule, report it honestly; don't tune on the test set.
