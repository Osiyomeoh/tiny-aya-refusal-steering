# Steering Tiny Aya away from unsafe electrical advice

Lexsi AI Safety and Alignment Hackathon. Track 2b, problem 8: inference-time safety steering.

We extract a refusal direction from `CohereLabs/tiny-aya-global` with SafeTune and add it at inference time, at a fixed strength of 0.25. There is no fine-tuning and the weights do not change. We then measure the steered build against the base build on 30 paired electrical-site-safety prompts and on AuditKIT's red-team probes.

Demo: https://tiny-aya-steering.vercel.app · Dataset: https://huggingface.co/datasets/osiyomeoh/tiny-aya-steering-electrical-safety · Steering direction: https://huggingface.co/osiyomeoh/tiny-aya-refusal-direction · HF model collection: https://huggingface.co/collections/osiyomeoh/tiny-aya-refusal-steering-models-lexsi-hackathon-6af08bc168e089a2ac96e2a6 · HF dataset collection: https://huggingface.co/collections/osiyomeoh/tiny-aya-refusal-steering-datasets-lexsi-hackathon-6af08bc2b5a4044550e30b3a

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

## Why these choices
- **The problem.** On an electrical site, the dangerous answer is a confident how-to: defeating a lockout, jumpering a safety relay, skipping the test for dead. The same worker also asks real questions about the same equipment. A fix has to cut the harmful how-to without refusing the legitimate question next to it.
- **The model.** Tiny Aya Global is on the hackathon's model list. Its [model card](https://huggingface.co/CohereLabs/tiny-aya-global) gives 3.35B parameters and 70+ training languages, and names "local deployment under realistic compute constraints" as a goal. A site team could run a model like this on its own hardware, so we ran it in NF4 on a single Colab T4 to match that constraint. We tested English only.
- **The method.** Inference-time steering changes no weights. The fix is one vector of 2,048 numbers (a 10 KB file) that can be switched on, switched off or turned up without retraining or re-uploading the model. It needs 64 calibration prompts, not a training set. Track 2b problem 8 asks for this: block harmful requests with no retraining while keeping over-refusal low.
- **The data.** We wrote 30 pairs so that every unsafe request has a legitimate twin on the same task. One prompt set then measures both harm and over-refusal on the same topics. A generic refusal benchmark would not show whether the model still helps electricians.
- **The metrics.**
  1. *Unsafe answers to unsafe requests*, labelled blind by two labellers and tested with a paired exact McNemar test: the harm we want to reduce. The test is paired because both builds answer the same prompts.
  2. *Legitimate questions refused*: the cost. A model that refused everything would score perfectly on the first metric, so this guards against that.
  3. *AuditKIT red-team pass/fail*: an independent tool from the track, to check whether steering broke or fixed anything outside our prompts (injection, jailbreak, encoding, over-refusal).

  The decision rule was fixed before the final run, so the result could not be tuned to pass.
- **What it changes in use.** An assistant on the steered model gave an unsafe procedure 5 times in 30 instead of 11, and still answered all 30 legitimate questions. Turning steering on needs no retraining but costs 15% of generation speed (see below). It still gives 5 unsafe answers in 30, so it is one layer of defence, not a guarantee.

## Cost compared with closed models
`spike/roi.py` computes this from the final run's timings. The GPU price is a [Hugging Face Inference Endpoints](https://huggingface.co/docs/inference-endpoints/pricing) T4 at $0.50 an hour. API prices are [Anthropic's list prices](https://platform.claude.com/docs/en/about-claude/pricing). Both were checked on Nov 8, 2026. The volume is an assumption: 200 electricians asking 5 questions each working day, 22,000 a month.

Measured at batch size 1: the steered build took 31.7 s and produced 358 tokens per answer, 11.3 tokens/s. The base build ran at 13.3 tokens/s, so the steering hooks cost 15% of throughput.

| Option | Cost per answer | Cost for 22,000 answers a month |
|---|---|---|
| Steered Tiny Aya on a T4 endpoint, GPU-hours used only (lower bound) | $0.00440 | $96.90 (194 GPU-hours) |
| Steered Tiny Aya on a T4 endpoint, always on | — | $365.00 |
| Claude Haiku 5.5 API | $0.00018 | $3.99 |
| Claude Sonnet 5.5 API | $0.00363 | $79.77 |
| Claude Opus 5.5 API | $0.00725 | $159.54 |

At batch size 1 on one T4, self-hosting is cheaper than Claude Opus 5.5 per answer. An always-on T4 breaks even with Opus at about 50,000 answers a month, within its capacity of about 83,000. It is dearer than Sonnet 5.5 and much dearer than Haiku 5.5. The case for running this model yourself is control rather than price: prompts stay on your own hardware, it can run without internet access, the safety fix is a file you can inspect, and the cost is fixed.

What this does not cover:
- Batched serving (for example vLLM) would raise throughput. We did not measure it, so every row uses batch size 1.
- We did not run the closed models on our prompts. This compares cost, not safety or quality.
- Every row uses Tiny Aya's token counts; closed models tokenize differently. The prompt length (24 tokens) is estimated from word counts.
- Tiny Aya Global is licensed CC-BY-NC. Commercial use needs a licence from Cohere, and its cost is not included.

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
7. Cost table: `python3 spike/roi.py`.

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
- Hugging Face model collection (public, ungated): https://huggingface.co/collections/osiyomeoh/tiny-aya-refusal-steering-models-lexsi-hackathon-6af08bc168e089a2ac96e2a6
- Hugging Face dataset collection (public, ungated): https://huggingface.co/collections/osiyomeoh/tiny-aya-refusal-steering-datasets-lexsi-hackathon-6af08bc2b5a4044550e30b3a

## Layout
- `spike/`: run cells, outputs, labels, scorer and tests, the cost script, plus the decision rules and their amendments.
- `site/`: the demo page and its data builder.
- `CLAIMS.md`: every number, with its source. `PLAN.md` and `CHECKLIST.md` track the plan and the checks.
