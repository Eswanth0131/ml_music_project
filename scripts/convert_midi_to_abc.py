import os
import subprocess
from multiprocessing import Pool, cpu_count
from pathlib import Path
import time
from tqdm import tqdm


MIDI_ROOT = "/Users/eswanthsriramchengalasetty/NYU/ML/data/raw_midi/lmd_full"
ABC_ROOT = "/Users/eswanthsriramchengalasetty/NYU/ML/data/abc"
NUM_WORKERS = max(1, int(cpu_count() * 0.65))
RETRY_LIMIT = 2
SCAN_BATCH_SIZE = 5000   
TEMP_SLEEP = 0.05        

os.makedirs(ABC_ROOT, exist_ok=True)

def convert_one(midi_file):
    try:
        relative_path = os.path.relpath(midi_file, MIDI_ROOT)
        abc_file = os.path.join(ABC_ROOT, Path(relative_path).with_suffix(".abc"))

        if os.path.exists(abc_file) and os.path.getsize(abc_file) > 0:
            return "SKIP"

        os.makedirs(os.path.dirname(abc_file), exist_ok=True)

        for _ in range(RETRY_LIMIT):
            try:
                result = subprocess.run(
                    ["midi2abc", midi_file],
                    capture_output=True,
                    text=True,
                    timeout=20
                )

                if result.returncode != 0:
                    continue

                with open(abc_file, "w") as f:
                    f.write(result.stdout)

                return "OK"

            except subprocess.TimeoutExpired:
                time.sleep(0.1)
                continue

        return "FAIL"

    except Exception:
        return "ERROR"


def batch_iterable(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def scan_midi_files():
    for root, _, files in os.walk(MIDI_ROOT):
        paths = [
            os.path.join(root, f)
            for f in files
            if f.lower().endswith((".mid", ".midi"))
        ]
        for chunk in batch_iterable(paths, SCAN_BATCH_SIZE):
            yield chunk


def count_all_files():
    count = 0
    for root, _, files in os.walk(MIDI_ROOT):
        count += sum(f.lower().endswith((".mid", ".midi")) for f in files)
    return count


def main():
    total_files = count_all_files()
    print(f"Total MIDI files: {total_files}")
    print(f"Using {NUM_WORKERS} workers")

    processed = 0

    with Pool(NUM_WORKERS) as pool, tqdm(total=total_files, unit="file") as pbar:
        for batch in scan_midi_files():
            for result in pool.imap_unordered(convert_one, batch):
                processed += 1
                pbar.update(1)
            time.sleep(TEMP_SLEEP)


if __name__ == "__main__":
    main()
