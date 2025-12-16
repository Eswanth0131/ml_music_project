import os
import json
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

ABC_CLEAN_DIR = "data/abc_clean"
VOCAB_PATH = "data/vocab.json"

OUT_TRAIN = "data/train.jsonl"
OUT_VAL = "data/val.jsonl"
OUT_TEST = "data/test.jsonl"

VAL_RATIO = 0.1
TEST_RATIO = 0.05
NUM_PROCESSES = 6


def tokenize_text(text, token_to_id):
    return [token_to_id[ch] for ch in text if ch in token_to_id]


def process_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        return path, txt
    except:
        return path, None


def main():
    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)
    token_to_id = vocab["token_to_id"]

    abc_files = [
        os.path.join(ABC_CLEAN_DIR, f)
        for f in os.listdir(ABC_CLEAN_DIR)
        if f.endswith(".abc")
    ]

    random.shuffle(abc_files)

    N = len(abc_files)
    n_val = int(N * VAL_RATIO)
    n_test = int(N * TEST_RATIO)

    val_files = set(abc_files[:n_val])
    test_files = set(abc_files[n_val:n_val+n_test])
    train_files = set(abc_files[n_val+n_test:])

    print(f"Total: {N}, Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    train_fp = open(OUT_TRAIN, "w")
    val_fp = open(OUT_VAL, "w")
    test_fp = open(OUT_TEST, "w")

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as ex:
        for path, txt in tqdm(ex.map(process_file, abc_files), total=N):
            if txt is None:
                continue

            tokens = tokenize_text(txt, token_to_id)
            record = json.dumps({"tokens": tokens}) + "\n"

            if path in train_files:
                train_fp.write(record)
            elif path in val_files:
                val_fp.write(record)
            else:
                test_fp.write(record)

    train_fp.close()
    val_fp.close()
    test_fp.close()

    print(f"  Train -> {OUT_TRAIN}")
    print(f"  Val   -> {OUT_VAL}")
    print(f"  Test  -> {OUT_TEST}")


if __name__ == "__main__":
    main()
