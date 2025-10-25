import random, shutil
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()
src_root = Path(os.environ.get("src_kaggle"))  # folder containing full.class, flat.class
print(src_root)

dst_root = Path(os.path.join(os.getcwd(), "data", "raw", "pressure"))

split_ratio = (0.78, 0.12, 0.10)  # train, val, test
random_seed = 42
move_files = False  # set to True if you want to move instead of copy

# === Script ===
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_images(class_dir: Path):
    return [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]

def split_indices(n, ratios, seed=42):
    random.seed(seed)
    idx = list(range(n))
    random.shuffle(idx)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_test = n - n_train - n_val
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

classes = ["full.class", "flat.class"]

for c in classes:
    src_c = src_root / c
    if not src_c.exists():
        print(f"[ERROR] Missing folder: {src_c}")
        continue

    out_c = c.replace(".class", "")
    imgs = collect_images(src_c)
    n = len(imgs)
    if n == 0:
        print(f"[WARN] No images in {src_c}")
        continue

    train_idx, val_idx, test_idx = split_indices(n, split_ratio, seed=random_seed)
    splits = {
        "train": [imgs[i] for i in train_idx],
        "val": [imgs[i] for i in val_idx],
        "test": [imgs[i] for i in test_idx],
    }

    for split, files in splits.items():
        out_dir = dst_root / split / out_c
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in files:
            dest = out_dir / p.name
            if move_files:
                shutil.move(str(p), str(dest))
            else:
                shutil.copy2(str(p), str(dest))

    print(f"[OK] {c}: {n} → train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])}")

print("\n[DONE] Pressure dataset split complete!")
