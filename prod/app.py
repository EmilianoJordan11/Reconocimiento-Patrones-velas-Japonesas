"""
App web de inferencia — Semana 4 (prod/).

Detector de patrones de velas japonesas. El usuario sube una imagen, el modelo
(Faster R-CNN ResNet50-FPN entrenado en dev/) detecta los patrones presentes y
la app muestra:
  - la imagen de entrada con los bounding boxes dibujados,
  - una tabla con las detecciones (clase + score como barra de confianza),
  - un resumen de confianza por patrón y una leyenda de colores por clase.

Ejecutar con:  streamlit run prod/app.py

NOTA: este archivo es solo la CAPA DE PRESENTACIÓN. Toda la carga del modelo, el
preprocesamiento (idéntico a val/test) y el postprocesamiento viven en utils.py
y NO se modifican por motivos estéticos.
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
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Texto interpretativo por patrón (CONTENIDO ESTÁTICO, no sale del modelo)
# --------------------------------------------------------------------------- #
# Diccionario fijo {nombre_patrón: descripción educativa}. El modelo solo DETECTA
# el patrón; estas descripciones son estáticas, neutrales y con fines educativos.
# Las claves coinciden EXACTAMENTE con utils.CLASS_NAMES.
#
# Reglas de contenido aplicadas:
#   - No son pronósticos ni consejos de inversión.
#   - Redactadas en condicional ("suele asociarse a", "podría indicar"…).
#   - Siempre acompañadas del disclaimer DISCLAIMER_TEXT.
PATTERN_INFO = {
    "Bearish Engulfing": (
        "El patrón **Bearish Engulfing** (envolvente bajista) está formado por dos "
        "velas: una alcista pequeña seguida de una vela bajista cuyo cuerpo "
        "*envuelve* por completo al cuerpo de la anterior. En análisis técnico "
        "**suele interpretarse** como una posible señal de agotamiento de un "
        "movimiento alcista previo. Históricamente **se asocia con** un cambio en "
        "el equilibrio entre compradores y vendedores a favor de estos últimos."
    ),
    "Bearish Insidebar": (
        "El patrón **Bearish Inside Bar** (barra interior bajista) aparece cuando "
        "una vela queda contenida dentro del rango (máximo–mínimo) de la vela "
        "anterior, en un contexto bajista. **Podría indicar** una pausa o "
        "consolidación tras un movimiento descendente. **Suele interpretarse** "
        "como una fase de indecisión o acumulación de volatilidad antes de que el "
        "precio defina su próxima dirección."
    ),
    "Bullish Engulfing": (
        "El patrón **Bullish Engulfing** (envolvente alcista) está formado por una "
        "vela bajista pequeña seguida de una vela alcista cuyo cuerpo *envuelve* "
        "por completo al de la anterior. **Históricamente se interpreta como** una "
        "posible señal de agotamiento de un movimiento bajista previo y **suele "
        "asociarse a** un cambio en el equilibrio entre vendedores y compradores a "
        "favor de estos últimos."
    ),
    "Bullish Insidebar": (
        "El patrón **Bullish Inside Bar** (barra interior alcista) ocurre cuando "
        "una vela queda contenida dentro del rango de la vela anterior, en un "
        "contexto alcista. **Podría indicar** una pausa o consolidación dentro de "
        "un movimiento ascendente. **Suele interpretarse** como una fase de "
        "indecisión previa a la definición de la siguiente dirección del precio."
    ),
    "Hammer": (
        "El patrón **Hammer** (martillo) es una vela con un cuerpo pequeño en la "
        "parte superior y una mecha inferior larga, que aparece tras un movimiento "
        "descendente. **Suele interpretarse** como un posible rechazo de los "
        "precios bajos durante la sesión. Históricamente **se asocia con** una "
        "posible pérdida de fuerza del movimiento bajista, aunque por sí solo no "
        "confirma nada."
    ),
    "Inverted_Hammer": (
        "El patrón **Inverted Hammer** (martillo invertido) presenta un cuerpo "
        "pequeño en la parte inferior y una mecha superior larga, apareciendo tras "
        "un movimiento descendente. **Podría indicar** un intento de los "
        "compradores por empujar el precio al alza durante la sesión. **Suele "
        "interpretarse** como una posible señal de cautela sobre la continuidad de "
        "la tendencia bajista, normalmente a la espera de confirmación posterior."
    ),
}

# Disclaimer obligatorio, siempre visible junto al texto interpretativo.
DISCLAIMER_TEXT = (
    "ℹ️ **Información educativa sobre análisis técnico. No es recomendación de "
    "inversión. Los patrones no garantizan resultados.**"
)


# --------------------------------------------------------------------------- #
# Estilos (CSS custom) — solo presentación
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    """Inyecta CSS para una estética cohesiva, prolija y clara."""
    st.markdown(
        """
        <style>
        /* Ancho de contenido más cómodo y centrado */
        .block-container { padding-top: 2.2rem; max-width: 1200px; }

        /* Header tipo "hero" con degradé suave del color de acento */
        .hero {
            background: linear-gradient(135deg, #4F46E5 0%, #6366F1 60%, #818CF8 100%);
            color: #FFFFFF;
            padding: 1.6rem 1.8rem;
            border-radius: 0.9rem;
            margin-bottom: 1.4rem;
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.18);
        }
        .hero h1 { color: #FFFFFF; margin: 0 0 .3rem 0; font-size: 1.9rem; }
        .hero p  { color: #EEF0FF; margin: 0; font-size: 1.02rem; }

        /* Tarjetas para enmarcar las imágenes y secciones */
        .card {
            background: #FFFFFF;
            border: 1px solid #E6E9F2;
            border-radius: 0.9rem;
            padding: 1rem 1.1rem;
            box-shadow: 0 2px 10px rgba(30, 41, 59, 0.05);
        }
        .card h3 { margin-top: 0; }

        /* Chips de la leyenda de colores por clase */
        .legend { display: flex; flex-wrap: wrap; gap: .5rem; }
        .chip {
            display: inline-flex; align-items: center; gap: .45rem;
            padding: .28rem .7rem; border-radius: 999px;
            background: #F4F6FB; border: 1px solid #E6E9F2;
            font-size: .86rem; color: #1E293B;
        }
        .dot { width: .8rem; height: .8rem; border-radius: 50%; display: inline-block; }

        /* "Pill" del patrón principal */
        .top-pill {
            display: inline-block; padding: .5rem 1rem; border-radius: 999px;
            background: #EEF2FF; color: #3730A3; font-weight: 600;
            border: 1px solid #C7D2FE;
        }

        /* Footer discreto */
        .footer { color: #94A3B8; font-size: .82rem; text-align: center; margin-top: 2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def rgb_to_hex(rgb: tuple) -> str:
    """Convierte (r, g, b) a string hex #RRGGBB (para CSS de la leyenda)."""
    return "#%02X%02X%02X" % rgb


def draw_detections(base_img: Image.Image, detections: list) -> Image.Image:
    """
    Dibuja las cajas detectadas sobre una COPIA de la imagen 224x224.

    No altera el pipeline: opera sobre result["image"] que ya viene del
    preprocesamiento de utils. Solo cuestión visual.
    """
    img = base_img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = utils.CLASS_COLORS.get(det["class_name"], (255, 0, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{det['class_name']} {det['score']:.0%}"
        ty = max(0, y1 - 12)
        if font is not None:
            tb = draw.textbbox((x1, ty), label, font=font)
            draw.rectangle(tb, fill=color)
            draw.text((x1, ty), label, fill="white", font=font)
        else:
            draw.text((x1, ty), label, fill=color)
    return img


def class_legend_html() -> str:
    """Genera el HTML de la leyenda de colores por clase."""
    chips = []
    for name in utils.CLASS_NAMES:
        color = rgb_to_hex(utils.CLASS_COLORS.get(name, (255, 0, 0)))
        chips.append(
            f'<span class="chip"><span class="dot" style="background:{color}"></span>{name}</span>'
        )
    return '<div class="legend">' + "".join(chips) + "</div>"


# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #
def render_sidebar() -> float:
    st.sidebar.markdown("### ⚙️ Configuración")
    score_threshold = st.sidebar.slider(
        "Umbral de confianza",
        min_value=0.0,
        max_value=1.0,
        value=utils.DEFAULT_SCORE_THRESHOLD,
        step=0.05,
        help="Solo se muestran detecciones con score mayor o igual a este valor.",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Modelo**")
    st.sidebar.caption("Faster R-CNN · ResNet50-FPN")
    st.sidebar.markdown("**Clases (6)**")
    st.sidebar.markdown(class_legend_html(), unsafe_allow_html=True)
    return score_threshold


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
inject_css()

st.markdown(
    """
    <div class="hero">
        <h1>🕯️ Detector de Patrones de Velas Japonesas</h1>
        <p>Subí una imagen de un gráfico de velas y el modelo detectará los patrones
        presentes (Engulfing, Inside Bar, Hammer…) con su nivel de confianza.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

score_threshold = render_sidebar()

# Cargar el modelo (cacheado con @st.cache_resource en utils.load_model).
try:
    model, device = utils.load_model()
except Exception as exc:  # noqa: BLE001
    st.error(f"No se pudo cargar el modelo.\n\n{exc}")
    st.stop()

dev_label = "GPU (CUDA)" if device == "cuda" else "CPU"
st.success(f"Modelo cargado correctamente · dispositivo: **{dev_label}**")

uploaded = st.file_uploader(
    "Subí una imagen (PNG / JPG)",
    type=["png", "jpg", "jpeg"],
)

if uploaded is None:
    st.info("Esperando una imagen para analizar…")
    st.markdown(
        '<div class="footer">Semana 4 · App de inferencia · Faster R-CNN</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# --- Inferencia (pipeline sin cambios) --------------------------------------
image = Image.open(BytesIO(uploaded.read()))
with st.spinner("Analizando la imagen…"):
    result = utils.predict(model, device, image, score_threshold=score_threshold)

detections = result["detections"]
annotated = draw_detections(result["image"], detections)

# --- Imagen original vs anotada, lado a lado --------------------------------
col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown('<div class="card"><h3>Imagen ingresada</h3>', unsafe_allow_html=True)
    st.image(result["image"], caption="Entrada (preprocesada a 224×224)", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown('<div class="card"><h3>Patrones detectados</h3>', unsafe_allow_html=True)
    st.image(annotated, caption="Detecciones con bounding boxes", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Patrón principal -------------------------------------------------------
st.markdown("### Resultado")
if result["top"] is not None:
    top = result["top"]
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(
            f'<span class="top-pill">Patrón principal: {top["class_name"]}</span>',
            unsafe_allow_html=True,
        )
    with c2:
        st.metric("Confianza", f"{top['score']:.1%}")

    # --- Texto interpretativo (estático) del patrón principal ----------------
    info = PATTERN_INFO.get(top["class_name"])
    if info:
        st.markdown(f"**¿Qué significa «{top['class_name']}»?**")
        st.markdown(info)
else:
    st.warning(
        "No se detectó ningún patrón por encima del umbral seleccionado "
        f"({score_threshold:.0%}). Probá bajar el umbral en la barra lateral."
    )

# Disclaimer SIEMPRE visible tras analizar una imagen (haya o no detección).
st.info(DISCLAIMER_TEXT)

# --- Tabla de detecciones con barra de confianza ----------------------------
if detections:
    st.markdown("#### Detecciones")
    df = pd.DataFrame(
        [
            {
                "Patrón": d["class_name"],
                "Confianza": d["score"],
                "Caja [x1, y1, x2, y2]": str(d["box"]),
            }
            for d in detections
        ]
    )
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confianza": st.column_config.ProgressColumn(
                "Confianza",
                help="Score de confianza de la detección",
                format="percent",
                min_value=0.0,
                max_value=1.0,
            ),
        },
    )

# --- Resumen de confianza por clase -----------------------------------------
st.markdown("#### Confianza por patrón")
st.caption("Score máximo detectado para cada uno de los 6 patrones.")
scores_df = pd.DataFrame(
    {
        "Patrón": list(result["class_scores"].keys()),
        "Confianza máx.": list(result["class_scores"].values()),
    }
).set_index("Patrón")
st.bar_chart(scores_df, color="#4F46E5")

st.markdown(
    '<div class="footer">Semana 4 · App de inferencia · Faster R-CNN ResNet50-FPN</div>',
    unsafe_allow_html=True,
)
