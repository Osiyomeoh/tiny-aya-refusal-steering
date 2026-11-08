---
pretty_name: Tiny Aya refusal steering, electrical site safety
language:
- en
license: "[FILL: check the output terms of the CohereLabs/tiny-aya-global licence]"
task_categories:
- text-generation
tags:
- ai-safety
- activation-steering
- refusal
- red-teaming
size_categories:
- n<1K
configs:
- config_name: steering
  data_files: steering.jsonl
- config_name: auditkit
  data_files: auditkit.jsonl
---

# Tiny Aya refusal steering: electrical site safety

These are the outputs and blind labels from an inference-time steering experiment on [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global), made for the Lexsi AI Safety and Alignment Hackathon (Track 2b, problem 8). Code and full method: [FILL: GitHub URL].

## Configs

**`steering`** (180 rows): 30 prompt pairs × 2 prompt types × 3 builds.

| Field | Meaning |
|---|---|
| `build` | `4bit_base`, `4bit_steer` (strength 0.25, the fixed configuration), `4bit_steer_0.5` (exploratory) |
| `strength` | steering strength; `null` for base |
| `pair_id`, `kind` | pair 0–29; `unsafe` request or `legit` question on the same task |
| `prompt`, `response` | the input and the greedy-decoded output (up to 512 new tokens) |
| `new_tokens` | tokens generated; 512 means the answer was cut off |
| `label_F1`, `label_F2` | two blind labellers. Unsafe prompts are labelled `SAFE`/`UNSAFE`; legit questions `ANSWERED`/`REFUSED` |

**`auditkit`** (94 rows): AuditKIT 1.1.6 `RedTeamRunner` probes on the base build and on steering at 0.25. Each probe was run only with its mapped detectors. Fields: `build`, `probe`, `detector`, `prompt`, `output`, `passed`.

## Setup
The model is loaded at revision `af89d219b53ed9b13b8a4645f9c8028973510324` in NF4 4-bit with BF16 compute, on a Colab T4. The refusal direction comes from SafeTune 0.1.13 (`extract_refusal_direction`, layer 18, from 32 harmful and 32 harmless calibration prompts) and is applied in steer mode across all layers.

## Results
- Unsafe answers: 11/30 for base and 5/30 for steering at 0.25. 6 pairs became safer and 0 worse (exact McNemar p = 0.031), the same for both labellers.
- Legitimate questions refused: 0/30 for both builds.
- AuditKIT: every probe/detector pass count is identical between the two builds.

## Labelling
The answers were shuffled with seed 20261108 and their build names hidden. Two fresh model labellers then labelled them against a written rubric (`spike/RUBRIC.md` in the code repo). They disagree on 3 of 180 rows.

## Limits and intended use
- One domain (electrical site safety), English only.
- The prompts were written by the author and have not been reviewed by an electrician.
- The labellers are models, not domain experts.
- 23 answers are cut off at 512 tokens.

Some responses in the `unsafe` rows contain dangerous electrical instructions; that is what the experiment measures. This is for safety research only. Do not use it as electrical guidance or as training data for helpful-assistant models.
