---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
tags:
- ai-safety
- activation-steering
- refusal-direction
- safetune
---

# Refusal steering direction for Tiny Aya Global

This repo holds a single refusal direction (`direction.pt`) for [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global). It is not a fine-tuned model: the base weights are unchanged. At inference time, SafeTune adds this vector to the residual stream. It was made for the Lexsi AI Safety and Alignment Hackathon (Track 2b, problem 8).

- **Base model:** `CohereLabs/tiny-aya-global` at revision `af89d219b53ed9b13b8a4645f9c8028973510324`, loaded in NF4 4-bit with BF16 compute.
- **Extraction:** SafeTune 0.1.13, `extract_refusal_direction(..., RefusalDirectionConfig(select_directions=False))`, using 32 harmful and 32 harmless prompts from `refusal_prompt_pairs_large(32)`. Taken from layer 18.
- **File contents:** a dict with `direction`, `layer`, `model`, `revision`, `safetune` and `calibration`. `direction` is a float32 unit vector of length 2048 (norm 1.0). SHA-256 of `direction.pt`: `036bcf013764d50a49a41e9701e9a1b71cb05f5ff9d591d3b1e6720045a2cab8`

## Use

```python
import torch, safetune.steer as S
d = torch.load("direction.pt", weights_only=True)
steered = S.RefusalDirectionModel(model, direction=d["direction"], mode="steer", strength=0.25, layers=None)
with steered:
    out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
```

## Evaluation at strength 0.25
These results are on 30 paired electrical-site-safety prompts, labelled blind by two labellers:
- Unsafe answers fell from 11/30 to 5/30. 6 pairs became safer and 0 worse (exact McNemar p = 0.031).
- Legitimate questions refused: 0/30, the same as base.
- AuditKIT 1.1.6 red-team: no pass/fail changed on any probe.

Outputs and labels: [FILL: HF dataset URL]. Code: [FILL: GitHub URL].

## Limits
- The evidence is one domain, English only, with 30 pairs and model labellers.
- Strength 0.25 was the only configuration fixed in advance. Higher strengths start refusing legitimate questions.
- This direction makes refusals more likely, but it is not a safety guarantee.

Licence: CC-BY-NC-4.0, inherited from the base model. Cohere Labs' [acceptable use policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy) also applies.
