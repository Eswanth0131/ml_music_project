import os
import torch
from torch.nn import functional as F
import pickle
from model import GPT, GPTConfig
import numpy as np

MODEL_PATH = "gpt/best"
OUT_FILE = "generated_samples.txt"
MAX_TOKENS = 600
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PREFIXES = [
    "X: 10\nT: The Dublin Jig\nM: 6/8\nL: 1/8\nK: G",
    "X: 11\nT: The Rolling Wave\nM: 4/4\nL: 1/8\nK: D",
    "X: 12\nT: Sweethearts Waltz\nM: 3/4\nL: 1/8\nK: A",
    "X: 13\nT: Lament for the Lost\nM: 3/4\nL: 1/4\nK: Am",
    "X: 14\nT: The C Major March\nM: 2/4\nL: 1/8\nK: C",
    "X: 15\nT: The Bouncing Tune\nM: 4/4\nL: 1/8\nK: G\n|: G2 D2 E2 G2 |",
]

def load_model(model_path):
    ckpt_path = os.path.join(model_path, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    conf = GPTConfig(**checkpoint['model_args'])

    model = GPT(conf)
    state_dict = checkpoint['model']

    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    meta_path = 'data/music_project/meta.pkl'
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.pkl not found at {meta_path}")

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    if 'stoi' in meta and 'itos' in meta:
        stoi, itos = meta['stoi'], meta['itos']
    elif 'vocab_to_int' in meta and 'int_to_vocab' in meta:
        stoi = meta['vocab_to_int']
        itos = meta['int_to_vocab']
    else:
        raise KeyError(f"meta.pkl loaded but did not contain expected vocabulary keys. Available keys: {list(meta.keys())}")

    return model, stoi, itos

def generate_sample(model, stoi, itos, prompt_text=""):
    prompt_with_newline = prompt_text + "\n" if prompt_text else "\n"
    start_ids = [stoi.get(c, 0) for c in prompt_with_newline]

    x = torch.tensor([start_ids], dtype=torch.long, device=DEVICE)
    y = model.generate(x, MAX_TOKENS, temperature=0.9, top_k=200)

    tokens = y[0].tolist()
    generated_text = ''.join([itos[i] for i in tokens])
    return generated_text

def main():
    try:
        model, stoi, itos = load_model(MODEL_PATH)
    except Exception as e:
        print(f"CRITICAL ERROR during model/meta loading: {e}")
        return

    print(f"Generating samples using model from {MODEL_PATH} on {DEVICE}...")

    with open(OUT_FILE, "w", encoding='utf-8') as f:

        for i in range(1, len(PREFIXES) + 1):
            print(f"Generating Unconditional Sample {i}/{len(PREFIXES)}...")
            sample_text = generate_sample(model, stoi, itos, prompt_text="")
            f.write(f"X: {i}\nT: Unconditional Tune {i}\n{sample_text.strip()}\n\n")

        f.write("\n" + "=" * 60 + "\n\n")

        for i, prefix in enumerate(PREFIXES):
            print(f"Generating Conditional Sample {i+1}/{len(PREFIXES)}...")
            sample_text = generate_sample(model, stoi, itos, prompt_text=prefix)
            generated_continuation = sample_text.split(prefix)[-1].strip()
            full_output = f"{prefix}\n{generated_continuation}\n"

            f.write(f"Conditional Prompt {i+1}\n{prefix}\n")
            f.write(f"Generated Music\n{full_output.strip()}\n\n")

    print(f"\nAll samples generated and saved to {OUT_FILE}")

if __name__ == "__main__":
    main()
