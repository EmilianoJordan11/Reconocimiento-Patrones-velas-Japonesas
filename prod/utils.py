"""
Utilidades de inferencia para la app de producción (Semana 4).

Este módulo replica EXACTAMENTE el pipeline de validación/test del notebook de
entrenamiento (dev/02_model_training.ipynb) y expone tres responsabilidades:

  1. Cargar el modelo entrenado (Faster R-CNN ResNet50-FPN) de forma cacheada.
  2. Preprocesar la imagen de entrada idéntico a val/test.
  3. Ejecutar la inferencia de DETECCIÓN y postprocesar las salidas.

IMPORTANTE — El modelo NO es un clasificador de imagen única: es un detector de
objetos. Por cada imagen devuelve un conjunto de detecciones, cada una con su
bounding box, su clase (uno de los 6 patrones) y su score de confianza.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

# Streamlit es opcional a nivel import: así utils.py se puede importar también
# desde un script de prueba sin Streamlit instalado. Si no está, el decorador
# de cache se vuelve un no-op.
try:
    import streamlit as st

    cache_resource = st.cache_resource
except Exception:  # pragma: no cover - solo aplica fuera de Streamlit

    def cache_resource(func=None, **_kwargs):
        """Fallback no-op de @st.cache_resource cuando Streamlit no está."""
        if func is None:
            return lambda f: f
        return func


# --------------------------------------------------------------------------- #
# Configuración (idéntica al notebook de entrenamiento)
# --------------------------------------------------------------------------- #

# Tamaño al que se redimensiona la imagen (IMAGE_SIZE del notebook).
IMAGE_SIZE = 224

# Nombres de clase en el ORDEN exacto del data.yaml. El índice de esta lista
# corresponde al class_id YOLO (0..5). En el modelo Faster R-CNN las etiquetas
# están desplazadas +1 (el 0 queda reservado para el fondo), por eso al mostrar
# un nombre se usa CLASS_NAMES[label - 1].
CLASS_NAMES = [
    "Bearish Engulfing",
    "Bearish Insidebar",
    "Bullish Engulfing",
    "Bullish Insidebar",
    "Hammer",
    "Inverted_Hammer",
]

# El modelo se construyó con num_classes = 6 patrones + 1 fondo = 7.
NUM_CLASSES = len(CLASS_NAMES) + 1

# Ruta local del modelo (mismo archivo producido por el notebook).
_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
LOCAL_MODEL_PATH = ROOT / "dev" / "modelo.pth"

# URL de descarga remota del modelo. El archivo pesa ~158 MB y supera el límite
# de 100 MB de GitHub, además de que hostings como Streamlit Cloud no resuelven
# bien los punteros de Git LFS. Dejá acá un enlace de descarga DIRECTA
# (Google Drive con confirm, Hugging Face, release de GitHub, etc.) o seteá la
# variable de entorno MODEL_URL en el panel de secrets del hosting.
#   - Hugging Face:  https://huggingface.co/<usuario>/<repo>/resolve/main/modelo.pth
#   - GitHub Release: https://github.com/<owner>/<repo>/releases/download/<tag>/modelo.pth
MODEL_URL = os.environ.get("MODEL_URL", "")

# Umbral de confianza por defecto para considerar válida una detección.
DEFAULT_SCORE_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# Carga del modelo
# --------------------------------------------------------------------------- #

def _build_model() -> torch.nn.Module:
    """
    Reconstruye la arquitectura EXACTA usada en entrenamiento:
    Faster R-CNN con backbone ResNet50-FPN y la cabeza adaptada a 7 clases
    (6 patrones + fondo).

    Nota: se instancia SIN pesos preentrenados (weights=None) porque a
    continuación se cargan los pesos finos del state_dict de modelo.pth. Esto
    evita descargar los pesos de COCO innecesariamente en producción.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=NUM_CLASSES,
        box_nms_thresh=0.3,  # mismo valor que en build_faster_rcnn_model()
    )
    # Asegurar que la cabeza tenga exactamente la forma esperada por el
    # state_dict (equivale al reemplazo del box_predictor del notebook).
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def _ensure_model_file() -> Path:
    """
    Garantiza que el archivo de pesos esté disponible localmente.

    Estrategia (decidida para soportar tanto local como cloud):
      1. Si existe dev/modelo.pth localmente (repo clonado con Git LFS), se usa.
      2. Si no, y hay MODEL_URL configurada, se descarga a una caché local.
      3. Si no hay ninguna de las dos, se lanza un error claro y accionable.
    """
    if LOCAL_MODEL_PATH.exists() and LOCAL_MODEL_PATH.stat().st_size > 1_000_000:
        # > 1 MB descarta el caso del "puntero" de Git LFS sin resolver.
        return LOCAL_MODEL_PATH

    # Destino de la descarga remota (no se versiona).
    cache_path = _THIS_DIR / "modelo.pth"
    if cache_path.exists() and cache_path.stat().st_size > 1_000_000:
        return cache_path

    if not MODEL_URL:
        raise FileNotFoundError(
            "No se encontró 'dev/modelo.pth' localmente (o es un puntero de Git "
            "LFS sin resolver) y no hay 'MODEL_URL' configurada.\n"
            "Soluciones:\n"
            "  - Local: instalá Git LFS y corré 'git lfs pull' para traer el "
            "binario real del modelo.\n"
            "  - Cloud: definí la variable de entorno/secret MODEL_URL con un "
            "enlace de descarga directa al modelo (Hugging Face, GitHub "
            "Release o Google Drive)."
        )

    _download_file(MODEL_URL, cache_path)
    return cache_path


def _is_gdrive_url(url: str) -> bool:
    """True si la URL apunta a Google Drive (necesita gdown, no urllib)."""
    return "drive.google.com" in url or "docs.google.com" in url


def _download_file(url: str, dest: Path) -> None:
    """
    Descarga un archivo a `dest`, mostrando progreso en la UI de Streamlit.

    Para Google Drive se usa gdown, porque los archivos grandes (>100 MB)
    devuelven una página HTML de confirmación de antivirus que urllib bajaría
    como si fuese el binario, corrompiendo el modelo. gdown sortea ese paso.
    Para cualquier otra URL de descarga directa se usa urllib con barra de
    progreso.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    progress = None
    try:
        progress = st.progress(0.0, text="Descargando el modelo (~158 MB)...")
    except Exception:
        pass  # fuera de Streamlit, sin barra de progreso

    if _is_gdrive_url(url):
        # gdown no expone un hook de progreso fino; mostramos un estado
        # indeterminado y dejamos que imprima su propio progreso en logs.
        import gdown

        if progress is not None:
            progress.progress(0.05, text="Descargando el modelo desde Google Drive...")
        # fuzzy=True permite pasar el link de "compartir" completo, no solo el id.
        gdown.download(url, str(tmp), quiet=False, fuzzy=True)
    else:
        import urllib.request

        def _hook(block_num, block_size, total_size):
            if progress is not None and total_size > 0:
                frac = min(1.0, (block_num * block_size) / total_size)
                progress.progress(frac, text=f"Descargando el modelo... {frac:6.1%}")

        urllib.request.urlretrieve(url, tmp, reporthook=_hook)

    # Sanidad: si la descarga falló silenciosamente (p. ej. Drive devolvió un
    # HTML de error), el archivo será diminuto. Evitamos cachear basura.
    if not tmp.exists() or tmp.stat().st_size < 1_000_000:
        if tmp.exists():
            tmp.unlink()
        if progress is not None:
            progress.empty()
        raise RuntimeError(
            "La descarga del modelo falló o devolvió un archivo inválido "
            f"(<1 MB). Verificá que MODEL_URL sea un enlace de descarga válido "
            f"y que el archivo esté compartido públicamente.\nURL: {url}"
        )

    tmp.replace(dest)
    if progress is not None:
        progress.empty()


@cache_resource
def load_model(device: str | None = None):
    """
    Carga el modelo de detección con sus pesos finos, en modo eval.

    Decorado con @st.cache_resource para que el modelo se cargue una sola vez
    por sesión del servidor de Streamlit (evita recargar 158 MB en cada
    interacción del usuario).

    Returns:
        (model, device): el modelo listo para inferir y el dispositivo usado.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = _ensure_model_file()
    model = _build_model()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


# --------------------------------------------------------------------------- #
# Preprocesamiento (IDÉNTICO a val/test del notebook)
# --------------------------------------------------------------------------- #

def convert_to_rgb(img: Image.Image) -> Image.Image:
    """
    Asegura RGB. Si la imagen es RGBA, la compone sobre un fondo NEGRO usando
    el canal alpha como máscara (idéntico a convert_to_rgb del notebook, que es
    coherente con el preprocesamiento 'sin fondo' del dataset).
    """
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (0, 0, 0))
        background.paste(img, mask=img.split()[3])
        return background
    return img.convert("RGB") if img.mode != "RGB" else img


def preprocess_image(img: Image.Image):
    """
    Replica el __getitem__ del CandlestickDetectionDataset para val/test:

        Image -> convert_to_rgb -> resize((224,224)) -> F.to_tensor

    NO se aplica Normalize: Faster R-CNN normaliza internamente con su propio
    GeneralizedRCNNTransform (image_mean/std de COCO). Agregar un Normalize acá
    rompería la coherencia con el entrenamiento.

    Returns:
        (img_tensor, img_resized_pil):
            img_tensor       -> tensor [3, 224, 224] en [0, 1] para el modelo.
            img_resized_pil  -> la imagen RGB 224x224 (para dibujar las cajas
                                sobre el MISMO espacio de coordenadas).
    """
    img_rgb = convert_to_rgb(img)
    img_resized = img_rgb.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_tensor = F.to_tensor(img_resized)
    return img_tensor, img_resized


# --------------------------------------------------------------------------- #
# Inferencia + postprocesamiento
# --------------------------------------------------------------------------- #

@torch.no_grad()
def predict(model, device, img: Image.Image, score_threshold: float = DEFAULT_SCORE_THRESHOLD):
    """
    Ejecuta la detección sobre una imagen PIL y devuelve resultados limpios.

    Args:
        model: modelo cargado con load_model().
        device: 'cuda' o 'cpu'.
        img: imagen PIL de entrada (cualquier tamaño/modo).
        score_threshold: umbral mínimo de confianza para reportar una detección.

    Returns:
        dict con:
          - 'image': PIL RGB 224x224 ya preprocesada (para graficar encima).
          - 'detections': lista de dicts ordenada por score desc, cada uno con
                {'label_idx' (1..6), 'class_name', 'score', 'box' [x1,y1,x2,y2]}.
          - 'class_scores': dict {class_name: mejor_score} agregando por clase
                el score máximo entre las detecciones que pasan el umbral
                (0.0 si la clase no fue detectada). Útil para mostrar una vista
                tipo "confianza por patrón".
          - 'top': la detección de mayor score (o None si no hubo ninguna).
    """
    img_tensor, img_resized = preprocess_image(img)
    output = model([img_tensor.to(device)])[0]

    boxes = output["boxes"].cpu()
    labels = output["labels"].cpu()
    scores = output["scores"].cpu()

    detections = []
    for box, label, score in zip(boxes, labels, scores):
        s = float(score)
        if s < score_threshold:
            continue
        label_idx = int(label)  # 1..6 (0 = fondo, no aparece en outputs)
        # Mapeo robusto: nombre solo si el índice cae en el rango de patrones.
        if 1 <= label_idx <= len(CLASS_NAMES):
            class_name = CLASS_NAMES[label_idx - 1]
        else:
            class_name = f"desconocido({label_idx})"
        detections.append(
            {
                "label_idx": label_idx,
                "class_name": class_name,
                "score": s,
                "box": [round(float(c), 1) for c in box.tolist()],
            }
        )

    detections.sort(key=lambda d: d["score"], reverse=True)

    # Agregado por clase: score máximo por patrón (0 si no se detectó).
    class_scores = {name: 0.0 for name in CLASS_NAMES}
    for det in detections:
        name = det["class_name"]
        if name in class_scores and det["score"] > class_scores[name]:
            class_scores[name] = det["score"]

    return {
        "image": img_resized,
        "detections": detections,
        "class_scores": class_scores,
        "top": detections[0] if detections else None,
    }


# Paleta determinística para dibujar cajas por clase (6 colores distinguibles).
CLASS_COLORS = {
    "Bearish Engulfing": (220, 50, 50),
    "Bearish Insidebar": (255, 140, 0),
    "Bullish Engulfing": (40, 180, 60),
    "Bullish Insidebar": (0, 160, 200),
    "Hammer": (150, 80, 220),
    "Inverted_Hammer": (230, 200, 0),
}
