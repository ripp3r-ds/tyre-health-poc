import csv
import shutil
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()
    # === Paths ===
src_root = Path(os.environ.get("src_roboflow"))
dst_root = Path(os.path.join(os.getcwd(), "data", "raw", "condition"))
splits = {
    "train": "train",
    "valid": "val",
    "test": "test"
}
def get_label(row):
    normal = row.get("NORMAL_Tyres", "0").strip()
    bald   = row.get("BALD_Tyres", "0").strip()
    bad    = row.get("BAD_Tyres", "0").strip()
    if normal == "1":
        return "good"
    if bald == "1" or bad == "1":
        return "worn"
    return None

def process_split(split_name, dst_name):
    csv_path = src_root / split_name / "_classes.csv"
    if not csv_path.exists():
        print(f"[WARN] Missing {csv_path}")
        return

    print(f"[INFO] Processing {csv_path}")
    dst_split = dst_root / dst_name
    (dst_split / "good").mkdir(parents=True, exist_ok=True)
    (dst_split / "worn").mkdir(parents=True, exist_ok=True)

    # --- open + sniff delimiter ---
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        except csv.Error:
            # fallback to comma
            class _D: delimiter = ","; quotechar = '"'
            dialect = _D()

        reader = csv.DictReader(f, dialect=dialect)
        # Normalize/strip header names
        reader.fieldnames = [h.strip() for h in (reader.fieldnames or [])]
        print("[DEBUG] Detected delimiter:", repr(dialect.delimiter))
        print("[DEBUG] CSV headers:", reader.fieldnames)

        # find the filename column (case-insensitive)
        filename_col = None
        for h in reader.fieldnames:
            if h.strip().lower() == "filename":
                filename_col = h
                break
        if not filename_col:
            raise KeyError("No 'filename' column found — headers were: " + str(reader.fieldnames))

        for row in reader:
            # strip whitespace around every cell
            row = { (k.strip() if isinstance(k,str) else k) :
                    (v.strip() if isinstance(v,str) else v)
                    for k,v in row.items() }

            fname = row.get(filename_col)
            if not fname:
                continue

            src_img = src_root / split_name / fname
            if not src_img.exists():
                print(f"  [SKIP] Missing image: {src_img.name}")
                continue

            label = get_label(row)
            if not label:
                continue

            dst_img = dst_split / label / fname
            shutil.copy2(src_img, dst_img)

    print(f"[DONE] {split_name} → {dst_split}")

def main():
    for s, d in splits.items():
        process_split(s, d)

if __name__ == "__main__":
    main()
