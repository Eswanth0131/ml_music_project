import os
import json
from tqdm import tqdm

ABC_CLEAN_DIR = "data/abc_clean"
VOCAB_PATH = "data/vocab.json"

def main():
    abc_files = [
        os.path.join(ABC_CLEAN_DIR, f)
        for f in os.listdir(ABC_CLEAN_DIR)
        if f.endswith(".abc")
    ]

    print(f"Building vocab from {len(abc_files)} cleaned ABC files.")

    vocab_chars = set()

    for path in tqdm(abc_files):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            for ch in text:
                vocab_chars.add(ch)
        except:
            pass

    vocab_chars = sorted(list(vocab_chars))

    vocab = {
        "token_to_id": {ch: i for i, ch in enumerate(vocab_chars)},
        "id_to_token": {i: ch for i, ch in enumerate(vocab_chars)},
    }

    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"Saved vocab (size {len(vocab_chars))} -> {VOCAB_PATH}")

if __name__ == "__main__":
    main()
