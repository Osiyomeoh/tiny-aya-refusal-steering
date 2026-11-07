# Labeller-agreement kill test

Written on 2026-11-06, before anyone labelled with `RUBRIC.md`.

Question: on safety labels for small-model outputs, do labellers disagree enough to swamp the effects that safety evals try to measure?

## Setup
- **Items:** the 30 `4bit` responses to unsafe prompts in `outputs.jsonl`, shuffled with seed 20261107, in `agree_to_label.csv`. The key is in `agree_key.csv`.
- **Labellers:** labelled independently with `RUBRIC.md`, without seeing anyone else's labels.
  - **H:** Sam.
  - **A1, A2:** two fresh Claude agents, each seeing only the rubric and the sheet.
- **Earlier labels (context only, not part of the rule):** the earlier Claude labels gave κ 0.59 on these 30 items, with 6/30 disagreeing, using different rubric wording.

## Rule
Primary statistic: Cohen's κ between H and each agent. Take the higher of the two (H–A1 and H–A2). That is the more generous case for the agents, so a PROCEED here is conservative.

| Result | Condition | Action |
|---|---|---|
| **PROCEED** | Best H–agent κ ≤ 0.60, or 5 or more disagreements out of 30 | Build a labeller-agreement / eval-noise tool |
| **KILL** | Best H–agent κ ≥ 0.80 and A1–A2 κ ≥ 0.80 | Labels are reliable enough; report the null |
| **EXTEND** | Anything else | Add a second human or 60+ items before deciding |

A1–A2 κ is reported. If it is high while H–agent κ is low, that means the agents are consistent with each other but differ from the human, which is itself a finding.

## Known limits
- **One human:** with a single human labeller, H is not ground truth.
- **Small sample:** 30 items give a wide κ confidence interval. Report a bootstrap 95% CI (2,000 resamples, seed 0).
- **Rubric:** the rubric was written by Claude after seeing earlier disagreements. It may be tuned to those cases.
