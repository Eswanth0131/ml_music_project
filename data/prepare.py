import os
import json
import pickle
import numpy as np
from tqdm import tqdm

input_train = 'train.jsonl'
input_val = 'val.jsonl'
out_dir = 'data/music_project'
os.makedirs(out_dir, exist_ok=True)

def process_file_stream(filename, bin_filename):
    print(f"Processing {filename}...")
    
    total_size = os.path.getsize(filename)
    
    out_path = os.path.join(out_dir, bin_filename)
    max_token = 0
    token_count = 0
    
    with open(out_path, 'wb') as f_out:
        with open(filename, 'r') as f_in:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=bin_filename) as pbar:
                for line in f_in:
                    pbar.update(len(line))
                    
                    try:
                        data = json.loads(line)
                        tokens = data['input_ids']
                        
                        if not tokens: continue

                        current_max = max(tokens)
                        if current_max > max_token:
                            max_token = current_max
                            
                        arr = np.array(tokens, dtype=np.uint16)
                        f_out.write(arr.tobytes())
                        token_count += len(tokens)
                        
                    except (json.JSONDecodeError, KeyError):
                        continue
                        
    print(f"Saved {bin_filename}: {token_count / 1e6:.2f} Million tokens")
    return max_token

max_val = process_file_stream(input_val, 'val.bin')
max_train = process_file_stream(input_train, 'train.bin')

vocab_size = max(max_train, max_val) + 1
print(f"Calculated Vocab Size: {vocab_size}")

meta = {'vocab_size': int(vocab_size)}
with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"Saved meta.pkl to {out_dir}")
