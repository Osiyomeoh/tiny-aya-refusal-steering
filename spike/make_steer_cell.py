"""Write steer_cell.py: one self-contained Colab cell for the steering kill test (see STEER-RULE.md)."""
import json

from make_notebook import PAIRS

CELL = '''!pip uninstall -y -q torchao
%pip install -q -U "safetune>=0.1.6" transformers accelerate bitsandbytes
import getpass, json, os, sys, time
import torch, transformers, safetune
from huggingface_hub import login, whoami
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import safetune.steer as S
from safetune.runner.utils import refusal_prompt_pairs_large
assert torch.cuda.is_available(), "Switch the runtime to GPU first."
from google.colab import drive, userdata
try:
    drive.mount("/content/drive")  # Optional: lets a rerun resume after a disconnect.
    SAVE = "/content/drive/MyDrive/lexis_steer"
except Exception as e:
    print("Drive not mounted (", e, ") - saving on this machine instead")
    SAVE = "/content/lexis_steer"
os.makedirs(SAVE, exist_ok=True)
OUT = SAVE + "/steer_outputs.jsonl"
DIRECTION = SAVE + "/direction.pt"
print("saving to", SAVE)
print(sys.version, torch.__version__, transformers.__version__, safetune.__version__, torch.cuda.get_device_name(0))
try:
    whoami()
except Exception:
    try:
        login(token=userdata.get("HF_TOKEN"))  # Colab Secrets, if set.
    except Exception:
        login(token=getpass.getpass("Hugging Face read token: "))  # Never saved to the notebook.
PAIRS = __PAIRS__
MODEL, REVISION = "CohereLabs/tiny-aya-global", "af89d219b53ed9b13b8a4645f9c8028973510324"
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
model.eval()

harmful, harmless = refusal_prompt_pairs_large(32)
print("calibration", len(harmful), len(harmless))
t0 = time.time()
if os.path.exists(DIRECTION):
    saved = torch.load(DIRECTION)
    direction, layer = saved["direction"], saved["layer"]
else:
    # Same calls as RefusalDirectionTrainer(alpha=0.25).calibrate, minus the layer sweep (STEER-RULE.md amendment).
    direction, layer, _ = S.extract_refusal_direction(model, tok, harmful, harmless,
                                                      S.RefusalDirectionConfig(select_directions=False))
    torch.save({"direction": direction.cpu(), "layer": layer}, DIRECTION)
steered = S.RefusalDirectionModel(model, direction=direction, mode="steer", strength=0.25, layers=None)
print("calibrated in", round(time.time() - t0), "s | direction from layer", layer)

def answer(prompt):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
    t = time.time()
    with torch.inference_mode():
        gen = model.generate(**ids, max_new_tokens=200, do_sample=False)
    new = gen[0][ids["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True), len(new), round(time.time() - t, 2)

done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        done = {(r["build"], r["pair_id"], r["kind"]) for r in map(json.loads, f)}
print("resuming with", len(done), "rows already saved")
with open(OUT, "a") as f:
    for build in ("4bit_base", "4bit_steer"):
        for i, pair in enumerate(PAIRS):
            for kind, prompt in zip(("unsafe", "legit"), pair):
                if (build, str(i), kind) in done:
                    continue
                if build == "4bit_steer":
                    with steered:
                        text, n, s = answer(prompt)
                else:
                    text, n, s = answer(prompt)
                f.write(json.dumps({"build": build, "pair_id": str(i), "kind": kind, "prompt": prompt,
                                    "response": text, "new_tokens": n, "seconds": s,
                                    "safetune": safetune.__version__, "alpha": 0.25}) + "\\n")
                f.flush(); os.fsync(f.fileno())
            print(build, i + 1, "/", len(PAIRS))
from google.colab import files
files.download(OUT)
'''


def main():
    with open("steer_cell.py", "w", encoding="utf-8") as f:
        f.write(CELL.replace("__PAIRS__", json.dumps(PAIRS)))


if __name__ == "__main__":
    main()
