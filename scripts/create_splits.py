"""
Script de generación de splits estratificados train/val/test.

Genera tres CSV (train.csv, val.csv, test.csv) con las rutas relativas a las imágenes
y etiquetas del dataset procesado, manteniendo la proporción de clases en cada split
(muestreo estratificado).

Diseño clave:
- Semilla fija (SEED=42) → resultados reproducibles entre máquinas.
- Estratificación por clase → cada split mantiene la distribución de las 6 clases.
- Las rutas se guardan como strings POSIX relativos a `data/processed/dataset_no_background/`,
  así los CSV son portables y livianos (sí se versionan en git).

Uso:
    python scripts/create_splits.py
"""

from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuración global del script.
# ---------------------------------------------------------------------------
DATASET_ROOT = Path("data/processed/dataset_no_background")  # Dataset YA procesado (sin fondo).
OUTPUT_DIR = Path("data")                                      # Dónde se guardan los CSV.
SEED = 42                                                       # Semilla para reproducibilidad.
TRAIN_RATIO = 0.70                                              # 70% train
VAL_RATIO = 0.15                                                # 15% val
TEST_RATIO = 0.15                                               # 15% test


def read_label_class(label_path: Path) -> int:
    """Lee la clase principal (primer box) de un archivo de etiquetas en formato YOLO.

    Cada línea del archivo tiene formato: `class_id x_center y_center width height`.
    Nosotros usamos solo `class_id` porque tratamos el problema como clasificación,
    no detección.
    """
    text = label_path.read_text().strip()
    if not text:
        raise ValueError(f"Etiqueta vacía: {label_path}")
    # Primer token = class_id. Convertimos a int.
    return int(text.split()[0])


def collect_samples(dataset_root: Path):
    """Recorre el dataset procesado y arma una lista plana de muestras.

    Cada muestra es un dict con:
        - image: ruta relativa a la imagen (string POSIX)
        - label: ruta relativa al label (string POSIX)
        - class_id: id de la clase (int) para poder estratificar

    Recorre las 3 carpetas (train/valid/test) que vienen del dataset original
    y las junta en una sola lista, ya que luego vamos a regenerar los splits
    desde cero con nuestra propia estratificación.
    """
    samples = []
    for split in ["train", "valid", "test"]:
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"

        # Si falta alguna de las dos carpetas, salteamos este split.
        if not image_dir.exists() or not label_dir.exists():
            continue

        # `sorted()` garantiza orden determinístico → reproducibilidad.
        for image_path in sorted(image_dir.glob("*.png")):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue  # Imagen sin label → la ignoramos.
            class_id = read_label_class(label_path)
            samples.append({
                "image": f"{split}/images/{image_path.name}",
                "label": f"{split}/labels/{label_path.name}",
                "class_id": class_id,
            })
    return samples


def stratified_splits(samples, seed=SEED, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO):
    """Divide la lista de muestras en train/val/test manteniendo la proporción de clases.

    Algoritmo:
        1. Agrupamos las muestras por clase (defaultdict).
        2. Para cada clase: barajamos sus muestras con la misma semilla
           y cortamos según las proporciones.
        3. Esto garantiza que cada clase aparezca en las 3 particiones
           con la misma proporción que en el total.
        4. Finalmente barajamos cada split completo para que no queden
           todas las muestras de la misma clase juntas.
    """
    # RandomState con semilla fija → la misma división cada vez que se corre.
    rng = np.random.RandomState(seed)

    # Agrupamos muestras por su class_id.
    class_groups = defaultdict(list)
    for sample in samples:
        class_groups[sample["class_id"]].append(sample)

    train, val, test = [], [], []

    # Para cada clase, dividimos sus muestras según las proporciones globales.
    for class_id, items in class_groups.items():
        items = list(items)
        rng.shuffle(items)  # Mezclamos dentro de la clase.

        n = len(items)
        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))
        n_test = n - n_train - n_val  # Sobrante va a test (evita perder muestras por redondeo).

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    # Mezclamos cada split para que las clases queden intercaladas.
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def save_split(split, path: Path):
    """Guarda un split como CSV con dos columnas: `image` y `label`.

    No guardamos `class_id` en el CSV porque ya está implícito en el archivo
    de label (se lee desde el `.txt` al cargar la muestra en el Dataset).
    """
    df = pd.DataFrame(split)[["image", "label"]]
    df.to_csv(path, index=False)
    print(f"Guardado {path} ({len(df)} filas)")


if __name__ == "__main__":
    # 1) Recolectar todas las muestras del dataset procesado.
    samples = collect_samples(DATASET_ROOT)
    if not samples:
        raise SystemExit(f"No se encontraron muestras en {DATASET_ROOT}")

    # 2) Estratificar en train/val/test.
    train_split, val_split, test_split = stratified_splits(samples)

    # 3) Escribir los 3 CSV en data/.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_split(train_split, OUTPUT_DIR / "train.csv")
    save_split(val_split, OUTPUT_DIR / "val.csv")
    save_split(test_split, OUTPUT_DIR / "test.csv")

    # 4) Reportar conteos finales para verificar a ojo.
    counts = {
        "train": len(train_split),
        "val": len(val_split),
        "test": len(test_split),
    }
    print("Distribución final:", counts)
