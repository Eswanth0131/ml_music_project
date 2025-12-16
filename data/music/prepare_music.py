import os
import glob
import pickle
import numpy as np
from tqdm import tqdm

SOURCE_DIR = os.path.join(os.path.dirname(__file__), 'abc_clean')
INPUT_FILE = os.path.join(os.path.dirname(__file__), 'input.txt')

def prepare():
    print(f"Searching for .abc files")
    abc_files = glob.glob(os.path.join(SOURCE_DIR, '*.abc'))

    if not abc_files:
        print(f"ERROR: No .abc files found")
        return

    print(f"Found {len(abc_files)} files")

    all_data = []
    for f_path in tqdm(abc_files):
        try:
            with open(f_path, 'r', encoding='utf-8', errors='ignore') as f:
                all_data.append(f.read() + "\n")
        except Exception as e:
            print(f"Skipping {f_path}: {e}")

    data = "".join(all_data)

    with open(INPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(data)

    print("Successfully merged into input.txt")
    print(f"Total dataset size: {len(data):,} characters")

    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size:,}")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    n = len(data)
    train_data = data[:int(n * 0.98)]
    val_data = data[int(n * 0.98):int(n * 0.99)]
    test_data = data[int(n * 0.99):]

    print(f"Train split: {len(train_data):,} tokens")
    print(f"Val split:   {len(val_data):,} tokens")

    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)
    test_ids = np.array(encode(test_data), dtype=np.uint16)

    base = os.path.dirname(__file__)
    train_ids.tofile(os.path.join(base, 'train.bin'))
    val_ids.tofile(os.path.join(base, 'val.bin'))
    test_ids.tofile(os.path.join(base, 'test.bin'))

    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }

    with open(os.path.join(base, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print("Done, ready for training!")

if __name__ == '__main__':
    prepare()
