import os
import pickle

INPUT_FILE = "input.txt"
META_FILE = "meta.pkl"

file_size = os.path.getsize(INPUT_FILE)

chars = set()
bytes_read = 0
last_print = 0

with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        bytes_read += len(line.encode("utf-8", errors="ignore"))
        chars.update(line)


chars = sorted(chars)
vocab_size = len(chars)

print(f"Vocab size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

meta = {
    "vocab_size": vocab_size,
    "stoi": stoi,
    "itos": itos
}

with open(META_FILE, "wb") as f:
    pickle.dump(meta, f)

print("meta.pkl written successfully")
