"""Write ft_cell.py: one self-contained Colab cell for the fine-tune drift kill test (see FT-RULE.md)."""
import json

from make_notebook import PAIRS

CELL = '''%pip install -q -U transformers accelerate bitsandbytes peft datasets
import gc, getpass, json, os, random, sys, time
import torch, transformers, peft
from datasets import load_dataset
from huggingface_hub import HfApi, login, whoami
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup
assert torch.cuda.is_available(), "Switch the runtime to GPU first."
print(sys.version, torch.__version__, transformers.__version__, peft.__version__, torch.cuda.get_device_name(0))
try:
    whoami()
except Exception:
    login(token=getpass.getpass("Hugging Face read token: "))  # Never saved to the notebook.
PAIRS = __PAIRS__
MODEL, REVISION = "CohereLabs/tiny-aya-global", "af89d219b53ed9b13b8a4645f9c8028973510324"
DATA = "databricks/databricks-dolly-15k"
DATA_REV = HfApi().dataset_info(DATA).sha
print("dataset revision", DATA_REV)
random.seed(0); torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
model.print_trainable_parameters()

rows = [r for r in load_dataset(DATA, split="train", revision=DATA_REV) if not r["context"].strip()]
random.Random(0).shuffle(rows)
examples = []
for r in rows[:1000]:
    user = [{"role": "user", "content": r["instruction"]}]
    prompt = tok.apply_chat_template(user, add_generation_prompt=True, tokenize=False)
    full = tok.apply_chat_template(user + [{"role": "assistant", "content": r["response"]}], tokenize=False)
    assert full.startswith(prompt), "chat template: prompt is not a prefix of the full turn"
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    t = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
    ids = (p + t)[:512]
    labels = ([-100] * len(p) + t)[:512]
    if any(x != -100 for x in labels):
        examples.append((ids, labels))
print("training examples", len(examples), "| target tail:", repr(full[len(prompt):][-60:]))

ACCUM, LR = 8, 2e-4
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=LR)
steps = len(examples) // ACCUM
sched = get_linear_schedule_with_warmup(opt, 10, steps)
model.config.use_cache = False
model.train()
start, running = time.time(), 0.0
for n, (ids, labels) in enumerate(examples[: steps * ACCUM]):
    out = model(input_ids=torch.tensor([ids], device=model.device), labels=torch.tensor([labels], device=model.device))
    (out.loss / ACCUM).backward()
    running += out.loss.item()
    if (n + 1) % ACCUM == 0:
        torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        step = (n + 1) // ACCUM
        if step % 10 == 0 or step == steps:
            print(f"step {step}/{steps} loss {running / (ACCUM * (10 if step % 10 == 0 else step % 10)):.3f} {time.time() - start:.0f}s")
            running = 0.0

model.eval(); model.config.use_cache = True
OUT = "ft_outputs.jsonl"
with open(OUT, "w") as f:
    for i, pair in enumerate(PAIRS):
        for kind, prompt in zip(("unsafe", "legit"), pair):
            ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True).to(model.device)
            t0 = time.time()
            with torch.inference_mode():
                gen = model.generate(**ids, max_new_tokens=200, do_sample=False)
            new = gen[0][ids["input_ids"].shape[1]:]
            f.write(json.dumps({"build": "4bit_ft", "pair_id": str(i), "kind": kind, "prompt": prompt,
                                "response": tok.decode(new, skip_special_tokens=True),
                                "new_tokens": len(new), "seconds": round(time.time() - t0, 2),
                                "dataset_revision": DATA_REV}) + "\\n")
            f.flush()
        print("ft", i + 1, "/", len(PAIRS))
from google.colab import files
files.download(OUT)
'''


def main():
    with open("ft_cell.py", "w", encoding="utf-8") as f:
        f.write(CELL.replace("__PAIRS__", json.dumps(PAIRS)))


if __name__ == "__main__":
    main()
