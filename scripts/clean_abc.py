import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

SOURCE_DIR = "data/abc"
OUT_DIR = "data/abc_clean"
os.makedirs(OUT_DIR, exist_ok=True)

def clean_abc_text(text):
    lines = text.splitlines()

    cleaned = []
    header_seen = False

    for line in lines:
        s = line.strip()

        if s.startswith("%"):  
            continue

        if not header_seen:
            if not s:  
                continue
            cleaned.append(s)
            if s.startswith("K:"):
                header_seen = True
            continue

        if not s:
            continue

        cleaned.append(s)

    final = []
    header_fields = set(["X:", "T:", "M:", "L:", "K:"])

    for line in cleaned:
        if any(line.startswith(h) for h in header_fields):
            final.append(line)
        else:
            break

    key_index = None
    for i, line in enumerate(cleaned):
        if line.startswith("K:"):
            key_index = i
            break

    if key_index is None:
        return None

    final = cleaned[:key_index + 1] + cleaned[key_index + 1:]

    return "\n".join(final) + "\n"


def process_file(path):
    fname = os.path.basename(path)
    out_path = os.path.join(OUT_DIR, fname)

    if os.path.exists(out_path):
        return True

    try:
        with open(path, "r", errors="ignore") as f:
            text = f.read()

        cleaned = clean_abc_text(text)
        if not cleaned:
            return False

        with open(out_path, "w") as f:
            f.write(cleaned)

        return True
    except:
        return False


def main():
    abc_files = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for f in files:
            if f.lower().endswith(".abc"):
                abc_files.append(os.path.join(root, f))

    print(f"Found {len(abc_files)} ABC files to clean.")

    success = 0
    fail = 0

    NUM_THREADS = 8  # safe + fast for M1/M2/M3

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as ex:
        futures = [ex.submit(process_file, f) for f in abc_files]

        for fut in tqdm(as_completed(futures), total=len(futures)):
            if fut.result():
                success += 1
            else:
                fail += 1


if __name__ == "__main__":
    main()
