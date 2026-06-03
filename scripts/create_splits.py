from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd


DATASET_ROOT = Path("data/processed/dataset_no_background")
OUTPUT_DIR = Path("data")
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def read_label_class(label_path: Path) -> int:
    text = label_path.read_text().strip()
    if not text:
        raise ValueError(f"Etiqueta vacía: {label_path}")
    return int(text.split()[0])


def collect_samples(dataset_root: Path):
    samples = []
    for split in ["train", "valid", "test"]:
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"
        if not image_dir.exists() or not label_dir.exists():
            continue
        for image_path in sorted(image_dir.glob("*.png")):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            class_id = read_label_class(label_path)
            samples.append({
                "image": f"{split}/images/{image_path.name}",
                "label": f"{split}/labels/{label_path.name}",
                "class_id": class_id,
            })
    return samples


def stratified_splits(samples, seed=SEED, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO):
    rng = np.random.RandomState(seed)
    class_groups = defaultdict(list)
    for sample in samples:
        class_groups[sample["class_id"]].append(sample)

    train, val, test = [], [], []
    for class_id, items in class_groups.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))
        n_test = n - n_train - n_val
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def save_split(split, path: Path):
    df = pd.DataFrame(split)[["image", "label"]]
    df.to_csv(path, index=False)
    print(f"Guardado {path} ({len(df)} filas)")


if __name__ == "__main__":
    samples = collect_samples(DATASET_ROOT)
    if not samples:
        raise SystemExit(f"No se encontraron muestras en {DATASET_ROOT}")

    train_split, val_split, test_split = stratified_splits(samples)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_split(train_split, OUTPUT_DIR / "train.csv")
    save_split(val_split, OUTPUT_DIR / "val.csv")
    save_split(test_split, OUTPUT_DIR / "test.csv")

    counts = {
        "train": len(train_split),
        "val": len(val_split),
        "test": len(test_split),
    }
    print("Distribución final:", counts)
