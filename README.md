# Steering Tiny Aya away from unsafe electrical advice

Lexsi AI Safety and Alignment Hackathon. Track 2b, problem 8: inference-time safety steering.

We extract a refusal direction from `CohereLabs/tiny-aya-global` with SafeTune and add it at inference time, at a fixed strength of 0.25. There is no fine-tuning and the weights do not change. We then measure the steered build against the base build on 30 paired electrical-site-safety prompts and on AuditKIT's red-team probes.

Demo: https://tiny-aya-steering.vercel.app · Dataset: [FILL: HF dataset collection] · Model notes: [FILL: HF model collection]

## Results

| Metric | Base | Steered 0.25 |
|---|---|---|
| Unsafe answers to 30 unsafe requests (both labellers) | 11/30 | 5/30 |
| Pairs made safer / made worse | — | 6 / 0 (exact McNemar p = 0.031, each labeller) |
| Legitimate questions refused (over-refusal) | 0/30 | 0/30 |
| AuditKIT prompt injection | 5/8 | 5/8 |
| AuditKIT jailbreak | 8/8 | 8/8 |
| AuditKIT encoding | 4/5 | 4/5 |
| AuditKIT over-refusal | 5/5 | 5/5 |

The decision rule was written down before the final run (`spike/STEER-RULE.md`). It says PROCEED if steering yields at least 5 more safer pairs than worse pairs and refuses no more than 3 of 30 legitimate questions. The result was a net of 6 with 0 refusals: **PROCEED**. Every claim, with its source file, is listed in [CLAIMS.md](CLAIMS.md).

On AuditKIT, steering changed the wording of 39 of 47 answers but did not change a single pass or fail. The refusal direction targets harmful requests, not prompt injection, so we report this as a null result.

Strength 0.5 was also run, for context only, and was not used in the decision. It gave 3/30 or 4/30 unsafe answers (depending on the labeller) but refused 1 legitimate question.

## How it was measured
- **Model:** `CohereLabs/tiny-aya-global` at revision `af89d219b53ed9b13b8a4645f9c8028973510324`. It was loaded in NF4 4-bit (bitsandbytes, double quantisation, BF16 compute) on a Colab T4, with greedy decoding and up to 512 new tokens.
- **Steering:** SafeTune 0.1.13. The direction comes from `S.extract_refusal_direction` with `RefusalDirectionConfig(select_directions=False)` using 32 harmful and 32 harmless calibration prompts from `refusal_prompt_pairs_large(32)`, and was taken from layer 18. `S.RefusalDirectionModel(mode="steer", strength=0.25)` applies it across all layers.
- **Prompts:** 30 pairs. Each pair has an unsafe request (for example, how to defeat a lockout) and a legitimate question on the same task. They are listed in `spike/final_cell.py`.
- **Labels:** the 180 answers (3 builds × 30 pairs × 2 prompts) were shuffled with seed 20261108 and labelled blind by two fresh model labellers against `spike/RUBRIC.md`. The labellers never saw the key. They disagree on 3 of 180 rows.
- **Statistics:** paired exact McNemar test on the unsafe prompts, computed separately for each labeller.
- **AuditKIT:** version 1.1.6, `RedTeamRunner`. Each probe was run with its mapped detectors only: injection, jailbreak and encoding use `keyword` + `injection_success`, and refusal uses `refusal`. The cell is `spike/audit_cell.py`.

## Reproduce
1. On Hugging Face, accept the licence for `CohereLabs/tiny-aya-global` and create a read token.
2. Paste `spike/final_cell.py` into a Colab cell with a T4 GPU runtime and run it. Enter the token when asked. It writes `steer_outputs.jsonl` (about 28 s per answer, roughly 85 minutes in total).
3. Build the blind labelling files: `cd spike && python3 make_steer_labels.py final`. Label `final_to_label.csv` against `RUBRIC.md` without opening `final_key.csv`.
4. Score the labels: `python3 steer_score.py final_key.csv final_labels_F1.csv`. The scorer has tests: `python3 -m unittest test_steer_score` (2 tests: rule boundaries and gain direction).
5. AuditKIT: run `spike/audit_cell.py` in Colab the same way. It writes `auditkit_redteam.jsonl`.
6. Demo: `python3 site/build_data.py`, then `cd site && python3 -m http.server`. The page reads only `data.json`.

The spike run (200 tokens) and the final run (512 tokens) began identically: all 120 final answers start with the spike answer word for word.

## Limits
- 30 pairs in one domain (electrical site safety), English only. I wrote the prompts myself and no electrician has reviewed them.
- The labellers are models applying a fixed rubric, not human experts.
- 23 of 180 answers hit the 512-token cap and are cut off.
- AuditKIT's detectors match keywords and patterns, and each probe has 5 to 8 prompts.
- One fixed strength was tested. We did not tune on the test set.

## Repos, models and tracks
- Track 2b, problem 8: inference-time safety steering.
- Model: [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global)
- Libraries: SafeTune 0.1.13 (PyPI `safetune`), AuditKIT 1.1.6 (PyPI `auditkit`).
- Code: https://github.com/Osiyomeoh/tiny-aya-refusal-steering

## Layout
- `spike/`: run cells, outputs, labels, scorer and tests, plus the decision rules and their amendments.
- `site/`: the demo page and its data builder.
- `CLAIMS.md`: every number, with its source. `PLAN.md` and `CHECKLIST.md` track the plan and the checks.
