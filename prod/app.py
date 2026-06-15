"""
App web de inferencia — Semana 4 (prod/).

Detector de patrones de velas japonesas. El usuario sube una imagen, el modelo
(Faster R-CNN ResNet50-FPN entrenado en dev/) detecta los patrones presentes y
la app muestra:
  - la imagen de entrada con los bounding boxes dibujados,
  - una tabla con las detecciones (clase + confianza + coordenadas),
  - un resumen de confianza por patrón.

Ejecutar con:  streamlit run prod/app.py
"""

from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

import utils

st.set_page_config(
    page_title="Detector de Patrones de Velas Japonesas",
    page_icon="🕯️",
    layout="wide",
)


def draw_detections(base_img: Image.Image, detections: list) -> Image.Image:
    """Dibuja las cajas detectadas sobre una copia de la imagen 224x224."""
    img = base_img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = utils.CLASS_COLORS.get(det["class_name"], (255, 0, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{det['class_name']} {det['score']:.2f}"
        # Fondo del texto para que sea legible sobre cualquier imagen.
        ty = max(0, y1 - 11)
        if font is not None:
            tb = draw.textbbox((x1, ty), label, font=font)
            draw.rectangle(tb, fill=color)
            draw.text((x1, ty), label, fill="white", font=font)
        else:
            draw.text((x1, ty), label, fill=color)
    return img


# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Configuración")
score_threshold = st.sidebar.slider(
    "Umbral de confianza",
    min_value=0.0,
    max_value=1.0,
    value=utils.DEFAULT_SCORE_THRESHOLD,
    step=0.05,
    help="Solo se muestran detecciones con score mayor o igual a este valor.",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Modelo:** Faster R-CNN ResNet50-FPN\n\n"
    "**Clases (6):**\n"
    + "\n".join(f"- {c}" for c in utils.CLASS_NAMES)
)

# --------------------------------------------------------------------------- #
# Cabecera
# --------------------------------------------------------------------------- #
st.title("🕯️ Detector de Patrones de Velas Japonesas")
st.caption(
    "Subí una imagen de un gráfico de velas y el modelo detectará los patrones "
    "presentes (Engulfing, Inside Bar, Hammer, etc.) con su nivel de confianza."
)

# Cargar el modelo (cacheado). Si falla, mostramos el mensaje accionable.
try:
    model, device = utils.load_model()
except Exception as exc:  # noqa: BLE001
    st.error(f"No se pudo cargar el modelo.\n\n{exc}")
    st.stop()

st.success(f"Modelo cargado correctamente (dispositivo: {device}).")

# --------------------------------------------------------------------------- #
# Entrada del usuario
# --------------------------------------------------------------------------- #
uploaded = st.file_uploader(
    "Subí una imagen (PNG / JPG)",
    type=["png", "jpg", "jpeg"],
)

if uploaded is None:
    st.info("Esperando una imagen para analizar...")
    st.stop()

image = Image.open(BytesIO(uploaded.read()))

with st.spinner("Analizando la imagen..."):
    result = utils.predict(model, device, image, score_threshold=score_threshold)

detections = result["detections"]
annotated = draw_detections(result["image"], detections)

# --------------------------------------------------------------------------- #
# Resultados
# --------------------------------------------------------------------------- #
col1, col2 = st.columns(2)
with col1:
    st.subheader("Imagen ingresada")
    st.image(result["image"], caption="Entrada (preprocesada a 224×224)", use_container_width=True)
with col2:
    st.subheader("Detecciones")
    st.image(annotated, caption="Patrones detectados", use_container_width=True)

# Patrón principal
st.markdown("---")
if result["top"] is not None:
    top = result["top"]
    st.subheader(f"Patrón principal: **{top['class_name']}**")
    st.metric("Confianza", f"{top['score']:.1%}")
else:
    st.warning(
        "No se detectó ningún patrón por encima del umbral seleccionado "
        f"({score_threshold:.0%}). Probá bajar el umbral en la barra lateral."
    )

# Tabla de detecciones
if detections:
    st.subheader("Tabla de detecciones")
    df = pd.DataFrame(
        [
            {
                "Patrón": d["class_name"],
                "Confianza": f"{d['score']:.1%}",
                "Caja [x1, y1, x2, y2]": str(d["box"]),
            }
            for d in detections
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

# Resumen de confianza por clase (score máximo por patrón)
st.subheader("Confianza por patrón")
scores_df = pd.DataFrame(
    {"Patrón": list(result["class_scores"].keys()),
     "Confianza máx.": list(result["class_scores"].values())}
).set_index("Patrón")
st.bar_chart(scores_df)
