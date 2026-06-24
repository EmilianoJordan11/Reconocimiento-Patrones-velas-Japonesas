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

import base64
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

import utils

st.set_page_config(
    page_title="Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ruta del logo, RELATIVA a este archivo (funciona también en Streamlit Cloud,
# donde no existen rutas absolutas de Windows). Si el archivo no está, la app
# sigue funcionando y muestra un fallback (ver render_hero).
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"


def get_logo_data_uri() -> str | None:
    """
    Devuelve el logo como data URI base64 para incrustarlo en el HTML del header.

    Se incrusta en base64 (en vez de una <img src="archivo">) porque el HTML
    inyectado con st.markdown no puede servir archivos locales por ruta. Si el
    archivo no existe o no se puede leer, devuelve None (fallback sin logo).
    """
    try:
        if LOGO_PATH.exists():
            data = LOGO_PATH.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            return f"data:image/png;base64,{b64}"
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# Texto interpretativo por patrón (CONTENIDO ESTÁTICO, no sale del modelo)
# --------------------------------------------------------------------------- #
# Diccionario fijo {nombre_patrón: info educativa}. El modelo solo DETECTA el
# patrón; estas descripciones son estáticas, neutrales y con fines educativos.
# Las claves coinciden EXACTAMENTE con utils.CLASS_NAMES.
#
# Cada entrada tiene:
#   - bias:    "alcista" | "bajista" (sesgo histórico del patrón).
#   - icon:    flecha 🔼 (posible alza) o 🔻 (posible baja).
#   - summary: frase corta en lenguaje natural que explica el gráfico y el sesgo.
#   - detail:  explicación más completa de qué es el patrón.
#
# Reglas de contenido aplicadas:
#   - NO son pronósticos ni consejos de inversión. El "sesgo" es la interpretación
#     HISTÓRICA del patrón, redactada en condicional ("suele asociarse a…").
#   - Siempre acompañadas del disclaimer DISCLAIMER_TEXT.
PATTERN_INFO = {
    "Bearish Engulfing": {
        "bias": "bajista",
        "icon": "🔻",
        "summary": (
            "En la imagen se ve un patrón **envolvente bajista**: una vela bajista "
            "grande que *envuelve* por completo a la vela alcista anterior. Eso es "
            "lo que ya ocurrió y forma el patrón. A partir de ahí, la teoría del "
            "análisis técnico lo **clasifica como bajista**: lo vigila como una "
            "posible señal de que el precio **podría continuar bajando** después."
        ),
        "detail": (
            "Está formado por dos velas: una alcista pequeña seguida de una vela "
            "bajista cuyo cuerpo envuelve por completo al de la anterior. Suele "
            "interpretarse como un posible agotamiento de la subida previa y un "
            "cambio de fuerza a favor de los vendedores."
        ),
    },
    "Bearish Insidebar": {
        "bias": "bajista",
        "icon": "🔻",
        "summary": (
            "En la imagen se ve una **barra interior bajista**: una vela chica "
            "queda contenida dentro de la vela anterior, dentro de un contexto "
            "bajista. Esa pausa es lo que forma el patrón. La teoría lo "
            "**clasifica como bajista**: la vigila como una posible señal de que el "
            "precio **podría seguir bajando** una vez resuelta la pausa."
        ),
        "detail": (
            "Aparece cuando una vela queda dentro del rango (máximo–mínimo) de la "
            "vela previa. Refleja indecisión o acumulación de volatilidad antes de "
            "que el precio defina su próxima dirección."
        ),
    },
    "Bullish Engulfing": {
        "bias": "alcista",
        "icon": "🔼",
        "summary": (
            "En la imagen se ve un patrón **envolvente alcista**: una vela alcista "
            "grande que *envuelve* por completo a la vela bajista anterior. Eso es "
            "lo que ya ocurrió y forma el patrón. A partir de ahí, la teoría lo "
            "**clasifica como alcista**: lo vigila como una posible señal de que el "
            "precio **podría continuar subiendo** después."
        ),
        "detail": (
            "Está formado por dos velas: una bajista pequeña seguida de una vela "
            "alcista cuyo cuerpo envuelve por completo al de la anterior. Suele "
            "interpretarse como un posible agotamiento de la baja previa y un "
            "cambio de fuerza a favor de los compradores."
        ),
    },
    "Bullish Insidebar": {
        "bias": "alcista",
        "icon": "🔼",
        "summary": (
            "En la imagen se ve una **barra interior alcista**: una vela chica "
            "queda contenida dentro de la vela anterior, dentro de un contexto "
            "alcista. Esa pausa es lo que forma el patrón. La teoría lo "
            "**clasifica como alcista**: la vigila como una posible señal de que el "
            "precio **podría seguir subiendo** una vez resuelta la pausa."
        ),
        "detail": (
            "Ocurre cuando una vela queda dentro del rango de la vela previa. "
            "Refleja una fase de indecisión previa a la definición de la siguiente "
            "dirección del precio."
        ),
    },
    "Hammer": {
        "bias": "alcista",
        "icon": "🔼",
        "summary": (
            "En la imagen se ve un **martillo**: una vela con cuerpo chico arriba y "
            "una mecha inferior larga, tras una caída. Esa mecha (el precio bajó "
            "pero se recuperó) es lo que forma el patrón. La teoría lo **clasifica "
            "como alcista**: lo vigila como una posible señal de que la baja "
            "**podría estar perdiendo fuerza** y girar al alza después."
        ),
        "detail": (
            "La mecha inferior larga indica que el precio bajó fuerte pero terminó "
            "recuperándose dentro de la sesión. Suele asociarse a una posible "
            "pérdida de fuerza del movimiento bajista, aunque por sí solo no "
            "confirma nada."
        ),
    },
    "Inverted_Hammer": {
        "bias": "alcista",
        "icon": "🔼",
        "summary": (
            "En la imagen se ve un **martillo invertido**: cuerpo chico abajo y una "
            "mecha superior larga, tras una caída. Ese intento de subida es lo que "
            "forma el patrón. La teoría lo **clasifica como alcista (de cautela)**: "
            "lo vigila como una posible señal de giro al **alza después**, "
            "normalmente a la espera de confirmación."
        ),
        "detail": (
            "La mecha superior larga muestra un intento de los compradores por "
            "empujar el precio hacia arriba. Suele interpretarse como una posible "
            "señal de cautela sobre la continuidad de la baja, a la espera de "
            "confirmación posterior."
        ),
    },
}

# Disclaimer obligatorio, siempre visible junto al texto interpretativo.
DISCLAIMER_TEXT = (
    "ℹ️ **Información educativa sobre análisis técnico. No es recomendación de "
    "inversión. Los patrones no garantizan resultados.**"
)
# Misma frase pero con HTML (para incrustarla en el <div> del disclaimer, donde
# el Markdown ** no se renderiza).
DISCLAIMER_HTML = (
    "ℹ️ <b>Información educativa sobre análisis técnico. No es recomendación de "
    "inversión. Los patrones no garantizan resultados.</b>"
)


# --------------------------------------------------------------------------- #
# Acondicionamiento opcional de la imagen de entrada (quitar fondo claro)
# --------------------------------------------------------------------------- #
# El modelo se entrenó con el dataset "no_background" (velas sobre fondo negro).
# Las capturas reales suelen tener fondo blanco/grilla, fuera de ese dominio, y
# el modelo no las reconoce. Esta función REPLICA exactamente el algoritmo de
# remove_white_background de dev/01_dataset_preparation.ipynb (flood fill desde
# las esquinas con umbral 240 + operaciones morfológicas) para llevar la imagen
# subida al MISMO dominio con el que se entrenó.
#
# Es solo un acondicionamiento de la ENTRADA del usuario; NO altera el
# preprocesamiento de val/test de utils.preprocess_image, que se aplica igual
# después sobre la imagen resultante.
WHITE_THRESHOLD = 240
MORPH_KERNEL_SIZE = 3


def remove_white_background(img_bgr: np.ndarray, threshold=WHITE_THRESHOLD,
                            kernel_size=MORPH_KERNEL_SIZE) -> np.ndarray:
    """Devuelve la imagen BGRA con el fondo blanco vuelto transparente."""
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    white_mask = (np.all(img_rgb >= threshold, axis=2).astype(np.uint8) * 255)

    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood_img = white_mask.copy()
    for fy, fx in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        if white_mask[fy, fx] == 255:
            cv2.floodFill(flood_img, flood_mask, (fx, fy), 128,
                          loDiff=10, upDiff=10, flags=cv2.FLOODFILL_FIXED_RANGE)

    bg_mask = (flood_img == 128).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_DILATE, kernel)

    alpha = cv2.bitwise_not(bg_mask)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    _, alpha = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)

    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, alpha])


def strip_white_background(img: Image.Image) -> Image.Image:
    """Aplica remove_white_background a una imagen PIL y devuelve una PIL RGBA."""
    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    rgba = remove_white_background(bgr)
    return Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))


# --------------------------------------------------------------------------- #
# Estilos (CSS custom) — solo presentación
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    """
    Inyecta TODO el CSS propio de la app: tema CLARO, formal y sobrio.
    Toda la presentación vive acá; los controles (uploader, slider, checkbox)
    siguen siendo widgets st.* y solo se les da estilo.

    Paleta:
        --bg       #F4F6F9  fondo claro (blanco hueso)
        --surface  #FFFFFF  tarjetas / superficies
        --border   #E2E8F0  bordes sutiles
        --ink      #1F2937  texto principal (gris azulado oscuro)
        --muted    #64748B  texto secundario
        --accent   #1E3A8A  azul marino corporativo (acento principal)
        --up       #15803D  verde alcista (sobrio)
        --down     #B91C1C  rojo bajista (sobrio)
    Tipografía: Fraunces (serif, títulos) + Inter (sans, cuerpo) +
    JetBrains Mono (números/scores).
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=JetBrains+Mono:wght@500;600&display=swap');

        :root {
            /* Paleta clara, formal y sobria */
            --bg: #F4F6F9; --bg-2: #FFFFFF; --surface: #FFFFFF; --surface-2: #F1F4F9;
            --border: #E2E8F0; --border-strong: #CBD5E1;
            --ink: #1F2937; --ink-soft: #334155; --muted: #64748B;
            --accent: #1E3A8A; --accent-soft: #2563EB; --accent-dim: rgba(37,99,235,.08);
            --up: #15803D; --up-dim: rgba(21,128,61,.09);
            --down: #B91C1C; --down-dim: rgba(185,28,28,.08);
            --shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px rgba(15,23,42,.06);
            --shadow-sm: 0 1px 3px rgba(15,23,42,.06);
        }

        /* Tipografía global: Inter para cuerpo, Fraunces (serif) para títulos */
        html, body, [class*="css"], .stApp, .stMarkdown, p, span, div, label,
        button, input, textarea { font-family: 'Inter', system-ui, sans-serif; }

        /* Ocultar el chrome de Streamlit (menú, Deploy, animación, footer) */
        #MainMenu { visibility: hidden; }
        header [data-testid="stToolbar"] { visibility: hidden; }
        [data-testid="stDecoration"] { display: none; }
        [data-testid="stStatusWidget"] { visibility: hidden; }
        header { background: transparent !important; }
        footer { visibility: hidden; }

        /* PERO mantener visible el botón que REABRE el sidebar cuando se colapsa. */
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"],
        [data-testid="stExpandSidebarButton"],
        [data-testid="stSidebarCollapseButton"] {
            visibility: visible !important;
            display: flex !important;
            z-index: 1000 !important;
        }

        .stApp { background:
            radial-gradient(1100px 520px at 85% -8%, rgba(37,99,235,.05), transparent 60%),
            radial-gradient(900px 480px at -8% 0%, rgba(30,58,138,.04), transparent 55%),
            var(--bg);
        }
        .block-container { padding-top: 1.8rem; padding-bottom: 3.5rem; max-width: 1120px; }

        h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.01em; }

        /* ===== HERO ======================================================== */
        .hero {
            position: relative; overflow: hidden;
            display: flex; align-items: center; gap: 1.5rem;
            background: linear-gradient(180deg, #FFFFFF 0%, #FAFBFD 100%);
            border: 1px solid var(--border);
            border-radius: 18px; padding: 1.7rem 2rem; margin-bottom: 1.6rem;
            box-shadow: var(--shadow);
        }
        /* Filete de acento superior */
        .hero::before {
            content: ""; position: absolute; left: 0; top: 0; right: 0; height: 4px;
            background: linear-gradient(90deg, var(--accent) 0%, var(--accent-soft) 100%);
        }
        .hero .logo-plate {
            z-index: 1; flex: 0 0 auto;
            background: #FFFFFF;
            border: 1px solid var(--border); border-radius: 16px;
            padding: .7rem .9rem; display: flex; align-items: center; justify-content: center;
            box-shadow: var(--shadow-sm);
        }
        .hero .logo { height: 76px; width: auto; object-fit: contain; display: block; }
        .hero .logo-fallback {
            height: 68px; width: 68px; border-radius: 16px; z-index: 1;
            display: flex; align-items: center; justify-content: center;
            background: var(--accent-dim); border: 1px solid var(--border); font-size: 2rem;
        }
        .hero-text { z-index: 1; }
        .hero-title { font-family: 'Fraunces', Georgia, serif; font-size: 2.1rem; font-weight: 600;
            color: var(--ink); margin: 0; line-height: 1.05; letter-spacing: -0.01em; }
        .hero-title .accent { color: var(--accent); }
        .hero-sub { color: var(--muted); font-size: 1.02rem; margin: .5rem 0 0 0; line-height: 1.5; max-width: 680px; }

        /* ===== FILA DE STATS / CHIPS ====================================== */
        .stats { display: flex; flex-wrap: wrap; gap: .8rem; margin-bottom: 1.8rem; }
        .stat {
            flex: 1 1 180px; background: var(--surface); border: 1px solid var(--border);
            border-radius: 12px; padding: .9rem 1.1rem; box-shadow: var(--shadow-sm);
            display: flex; align-items: center; gap: .8rem; transition: box-shadow .2s, transform .2s;
        }
        .stat:hover { box-shadow: var(--shadow); transform: translateY(-1px); }
        .stat .ico { font-size: 1.25rem; }
        .stat .k { color: var(--muted); font-size: .7rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }
        .stat .v { color: var(--ink); font-weight: 700; font-size: 1.02rem;
            font-family: 'JetBrains Mono', monospace; }
        .stat .v.green { color: var(--up); }

        /* ===== TARJETAS GENÉRICAS ========================================= */
        .card {
            background: var(--surface); border: 1px solid var(--border);
            border-radius: 14px; padding: 1.3rem 1.5rem; margin-bottom: 1.1rem;
            box-shadow: var(--shadow); transition: box-shadow .2s, transform .2s;
        }
        .card:hover { box-shadow: 0 2px 4px rgba(15,23,42,.05), 0 12px 32px rgba(15,23,42,.09); }
        .card-title {
            font-weight: 700; color: var(--ink); font-size: 1.02rem;
            margin: 0 0 .85rem 0; padding-bottom: .65rem;
            border-bottom: 1px solid var(--border);
            display: flex; align-items: center; gap: .5rem;
        }
        .section-title {
            font-family: 'Fraunces', Georgia, serif;
            font-weight: 600; color: var(--ink); font-size: 1.4rem;
            margin: 2rem 0 1rem 0; letter-spacing: -0.01em;
        }

        /* Imagen dentro de tarjeta */
        .img-card { padding: .9rem; }
        .img-card img { border-radius: 10px; width: 100%; display: block; border: 1px solid var(--border); }
        .img-card .cap { color: var(--muted); font-size: .82rem; text-align: center; margin-top: .7rem; }

        /* ===== ESTADO VACÍO =============================================== */
        .empty {
            text-align: center; padding: 3.2rem 1.5rem; margin-top: .5rem;
            background: var(--surface); border: 1.5px dashed var(--border-strong);
            border-radius: 16px; box-shadow: var(--shadow-sm);
        }
        .empty .big { font-size: 3.2rem; line-height: 1; margin-bottom: .7rem; }
        .empty .t { color: var(--ink); font-weight: 700; font-size: 1.18rem; }
        .empty .d { color: var(--muted); font-size: .95rem; margin-top: .4rem; }
        .empty .steps { display: flex; justify-content: center; gap: 1.6rem; flex-wrap: wrap; margin-top: 1.8rem; }
        .empty .step { display: flex; align-items: center; gap: .55rem; color: var(--ink-soft); font-size: .9rem; }
        .empty .step .n {
            width: 26px; height: 26px; border-radius: 50%; flex: 0 0 auto;
            display: flex; align-items: center; justify-content: center;
            background: var(--accent-dim); color: var(--accent-soft);
            font-weight: 700; font-size: .85rem; font-family: 'JetBrains Mono', monospace;
        }

        /* ===== BLOQUE DE SESGO (alza / baja) ============================== */
        .bias {
            display: flex; align-items: center; gap: .8rem;
            border-radius: 10px; padding: .9rem 1.1rem; font-weight: 600;
            border: 1px solid var(--border); margin-bottom: .8rem;
        }
        .bias .ico { font-size: 1.4rem; line-height: 1; }
        .bias.up   { background: var(--up-dim);   border-color: rgba(21,128,61,.28);  color: var(--up); }
        .bias.down { background: var(--down-dim); border-color: rgba(185,28,28,.28); color: var(--down); }

        .pattern-text { color: var(--ink-soft); line-height: 1.65; }
        .pattern-text b { color: var(--ink); }
        .note { color: var(--muted); font-size: .85rem; margin-top: .5rem; line-height: 1.55; }

        /* Tarjeta de explicación con borde de acento a la izquierda */
        .explain-card { border-left: 3px solid var(--accent); }

        /* ===== DISCLAIMER (siempre visible) =============================== */
        .disclaimer {
            background: var(--accent-dim); border: 1px solid rgba(37,99,235,.22);
            color: var(--accent); border-radius: 10px; padding: .9rem 1.1rem; font-size: .88rem;
            margin: 1rem 0; line-height: 1.55;
        }
        .disclaimer b { color: var(--accent); }

        /* ===== BARRAS DE CONFIANZA POR CLASE ============================== */
        .scores { display: flex; flex-direction: column; gap: .6rem; margin-top: .3rem; }
        .score-row { display: flex; align-items: center; gap: .8rem; }
        .score-name { width: 155px; font-size: .85rem; color: var(--ink-soft); flex: 0 0 auto; }
        .score-track {
            flex: 1; height: 9px; background: var(--surface-2); border-radius: 999px;
            overflow: hidden; border: 1px solid var(--border);
        }
        .score-fill { height: 100%; border-radius: 999px; transition: width .4s ease; }
        .score-val { width: 46px; text-align: right; font-size: .82rem; color: var(--muted);
            flex: 0 0 auto; font-family: 'JetBrains Mono', monospace; }

        /* ===== UPLOADER ESTILIZADO ======================================== */
        [data-testid="stFileUploaderDropzone"] {
            background: var(--surface); border: 1.5px dashed var(--border-strong);
            border-radius: 12px; transition: border-color .2s, background .2s;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--accent-soft); background: var(--accent-dim);
        }
        [data-testid="stFileUploaderDropzone"] * { color: var(--muted); }
        [data-testid="stFileUploaderDropzone"] button {
            background: var(--accent); color: #FFFFFF;
            border: 1px solid var(--accent); border-radius: 9px; font-weight: 600;
        }
        [data-testid="stFileUploaderDropzone"] button:hover {
            background: var(--accent-soft); border-color: var(--accent-soft);
        }

        /* ===== BOTONES ==================================================== */
        .stButton > button {
            border-radius: 9px; border: 1px solid var(--border-strong);
            background: var(--surface); color: var(--ink-soft); font-weight: 600;
            box-shadow: var(--shadow-sm);
            transition: border-color .2s, background .2s, color .2s, transform .1s;
        }
        .stButton > button:hover {
            border-color: var(--accent-soft); color: var(--accent); background: var(--accent-dim);
        }
        .stButton > button:active { transform: translateY(1px); }

        /* ===== BANNER "modelo cargado" (st.success) ====================== */
        [data-testid="stAlert"] { border-radius: 10px; }

        /* ===== TABLA / DATAFRAME ========================================= */
        [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; }

        /* ===== EXPANDERS ================================================= */
        [data-testid="stExpander"] {
            border: 1px solid var(--border); border-radius: 10px; background: var(--surface);
            box-shadow: var(--shadow-sm);
        }

        /* ===== BARRA LATERAL ============================================= */
        [data-testid="stSidebar"] {
            background: #FFFFFF; border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
        .side-title { font-family: 'Fraunces', Georgia, serif; font-size: 1.2rem; font-weight: 600;
            color: var(--ink); margin: .1rem 0 .9rem 0; }
        .side-help {
            background: var(--surface-2); border: 1px solid var(--border);
            border-radius: 10px; padding: .85rem .95rem; font-size: .85rem;
            color: var(--ink-soft); margin-bottom: 1.2rem; line-height: 1.5;
        }
        .side-help b { color: var(--accent); }
        .side-help ol { margin: .45rem 0 0 0; padding-left: 1.1rem; }
        .side-help li { margin-bottom: .25rem; }
        .side-label {
            font-size: .7rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: .06em; color: var(--muted); margin: 1.2rem 0 .45rem 0;
        }
        .side-foot { color: var(--muted); font-size: .74rem; margin-top: 1.6rem;
            border-top: 1px solid var(--border); padding-top: .85rem;
            font-family: 'JetBrains Mono', monospace; }

        /* ===== LEYENDA POR CLASE (agrupada por sesgo) ==================== */
        .legend-group { margin-bottom: .8rem; }
        .legend-head { font-size: .8rem; font-weight: 700; color: var(--ink); margin-bottom: .45rem; }
        .legend { display: flex; flex-direction: column; gap: .4rem; padding-left: .15rem; }
        .chip { display: inline-flex; align-items: center; gap: .55rem; font-size: .85rem; color: var(--ink-soft); }
        .dot  { width: .8rem; height: .8rem; border-radius: 50%; display: inline-block;
            flex: 0 0 auto; box-shadow: 0 0 0 2px rgba(15,23,42,.04); }

        /* ===== FOOTER ==================================================== */
        .footer {
            text-align: center; margin-top: 2.8rem; padding-top: 1.3rem;
            border-top: 1px solid var(--border); color: var(--muted); font-size: .85rem;
        }
        .footer .brand { color: var(--accent); font-weight: 700; }
        .footer .mono { font-family: 'JetBrains Mono', monospace; font-size: .78rem; color: var(--muted); }

        /* ===== RESPONSIVE =============================================== */
        @media (max-width: 640px) {
            .hero { flex-direction: column; text-align: center; gap: 1rem; }
            .hero-sub { font-size: .95rem; }
            .stat { flex: 1 1 100%; }
            .score-name { width: 110px; font-size: .8rem; }
            .empty .steps { flex-direction: column; align-items: center; gap: .7rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def rgb_to_hex(rgb: tuple) -> str:
    """Convierte (r, g, b) a string hex #RRGGBB (para CSS de la leyenda)."""
    return "#%02X%02X%02X" % rgb


def pil_to_data_uri(img: Image.Image) -> str:
    """Convierte una imagen PIL a data URI PNG para incrustarla en HTML."""
    buf = BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def md_inline_to_html(text: str) -> str:
    """
    Convierte el markdown inline (**negrita** y *cursiva*) a <b>/<i>, para poder
    incrustar los textos de PATTERN_INFO dentro de HTML propio. No interpreta más
    que esos dos casos a propósito (el contenido es controlado por nosotros).
    """
    import re

    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    return text


def image_card_html(img: Image.Image, title: str, caption: str) -> str:
    """HTML de una tarjeta que enmarca una imagen con título y epígrafe."""
    uri = pil_to_data_uri(img)
    return (
        f'<div class="card img-card">'
        f'<div class="card-title">{title}</div>'
        f'<img src="{uri}" alt="{title}" />'
        f'<div class="cap">{caption}</div>'
        f"</div>"
    )


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
    """
    Leyenda de colores por clase, AGRUPADA por sesgo (alcista / bajista) según
    PATTERN_INFO, para que de un vistazo se entienda de qué tipo es cada patrón.
    """
    groups = {"alcista": [], "bajista": []}
    for name in utils.CLASS_NAMES:
        info = PATTERN_INFO.get(name, {})
        bias = info.get("bias", "alcista")
        color = rgb_to_hex(utils.CLASS_COLORS.get(name, (255, 0, 0)))
        groups.setdefault(bias, []).append(
            f'<span class="chip"><span class="dot" style="background:{color}"></span>{name}</span>'
        )

    def block(title, icon, items):
        return (
            f'<div class="legend-group">'
            f'<div class="legend-head">{icon} {title}</div>'
            f'<div class="legend">' + "".join(items) + "</div>"
            f"</div>"
        )

    html = ""
    if groups.get("alcista"):
        html += block("Alcistas", "🔼", groups["alcista"])
    if groups.get("bajista"):
        html += block("Bajistas", "🔻", groups["bajista"])
    return html


def class_scores_html(class_scores: dict) -> str:
    """
    HTML de barras de confianza por clase (las 6), cada una con su color de clase.

    Se usa en vez de st.bar_chart porque cuando hay un solo patrón detectado el
    resto queda en 0 y el gráfico nativo se ve casi vacío. Las barras se ordenan
    por score descendente para que lo detectado quede arriba.
    """
    rows = sorted(class_scores.items(), key=lambda kv: kv[1], reverse=True)
    bars = []
    for name, score in rows:
        color = rgb_to_hex(utils.CLASS_COLORS.get(name, (148, 163, 184)))
        pct = max(0.0, min(1.0, float(score))) * 100
        # Barra "apagada" (gris) si el score es 0, para que se note que no se detectó.
        fill = color if score > 0 else "#E2E8F0"
        bars.append(
            f'<div class="score-row">'
            f'<div class="score-name">{name}</div>'
            f'<div class="score-track"><div class="score-fill" '
            f'style="width:{pct:.0f}%;background:{fill}"></div></div>'
            f'<div class="score-val">{pct:.0f}%</div>'
            f"</div>"
        )
    return '<div class="scores">' + "".join(bars) + "</div>"


def render_hero() -> str:
    """
    HTML del header "hero": logo + título + subtítulo sobre una banda con degradé
    oscuro→acento. Si no hay logo, usa un fallback con emoji (la app no se rompe).

    IMPORTANTE: el HTML se arma SIN saltos de línea ni indentación, porque
    st.markdown trata las líneas con 4+ espacios como bloque de código y mostraría
    las etiquetas como texto crudo.
    """
    logo_uri = get_logo_data_uri()
    if logo_uri:
        # El logo es azul oscuro: lo apoyamos sobre una baldosa clara para que
        # contraste con el fondo oscuro del hero y se lea bien.
        logo_html = (
            '<div class="logo-plate">'
            f'<img class="logo" src="{logo_uri}" alt="Predictor" />'
            "</div>"
        )
    else:
        logo_html = '<div class="logo-fallback">📈</div>'

    sub = (
        "Subí una imagen de un gráfico de velas y te explico, en palabras simples, "
        "qué patrón aparece y qué suele indicar."
    )
    return (
        '<div class="hero">'
        f"{logo_html}"
        '<div class="hero-text">'
        '<p class="hero-title">Predict<span class="accent">or</span></p>'
        f'<p class="hero-sub">{sub}</p>'
        "</div>"
        "</div>"
    )


def render_stats(model_ok: bool, dev_label: str, n_classes: int, threshold: float) -> str:
    """
    Fila de "stats" bajo el hero: estado del modelo, dispositivo, nº de clases y
    umbral actual. Tarjetitas pequeñas en una fila para llenar el espacio muerto.
    """
    estado = "Cargado" if model_ok else "Error"
    estado_cls = "green" if model_ok else ""

    def stat(ico, k, v, cls=""):
        return (
            f'<div class="stat"><span class="ico">{ico}</span>'
            f'<div><div class="k">{k}</div>'
            f'<div class="v {cls}">{v}</div></div></div>'
        )

    return (
        '<div class="stats">'
        + stat("🟢" if model_ok else "🔴", "Modelo", estado, estado_cls)
        + stat("⚡", "Dispositivo", dev_label)
        + stat("🎯", "Patrones", str(n_classes))
        + stat("📊", "Umbral", f"{threshold:.0%}")
        + "</div>"
    )


def render_empty_state() -> str:
    """Placeholder centrado para cuando no hay imagen subida (mata la pantalla vacía)."""
    return (
        '<div class="empty">'
        '<div class="big">🕯️</div>'
        '<div class="t">Subí un gráfico de velas para empezar</div>'
        '<div class="d">Arrastrá una imagen al recuadro de arriba o tocá "Browse files".</div>'
        '<div class="steps">'
        '<div class="step"><span class="n">1</span>Subí la imagen</div>'
        '<div class="step"><span class="n">2</span>El modelo detecta el patrón</div>'
        '<div class="step"><span class="n">3</span>Te explico qué significa</div>'
        "</div>"
        "</div>"
    )


def render_footer() -> str:
    """Footer cohesivo con la marca."""
    return (
        '<div class="footer">'
        '<span class="brand">Predictor</span> · análisis de patrones de velas japonesas<br>'
        '<span class="mono">Faster R-CNN · ResNet50-FPN</span>'
        "</div>"
    )


def limitations_html() -> str:
    """
    HTML con las limitaciones de uso de la app. Los números salen del dataset
    real de entrenamiento (ver dev/): IMAGE_SIZE=224 y ~1 patrón por imagen.
    """
    return (
        '<div class="card" style="border-left:4px solid var(--cool)">'
        '<div class="card-title">⚠️ Limitaciones de la app</div>'
        '<ul class="pattern-text" style="margin:0;padding-left:1.1rem;line-height:1.7">'
        "<li><b>Tamaño de imagen:</b> internamente toda imagen se redimensiona a "
        "<b>224×224 px</b>. Conviene subir imágenes aproximadamente cuadradas y "
        "nítidas; las muy alargadas o de baja resolución se deforman y se detectan peor.</li>"
        "<li><b>Cuántas velas por imagen:</b> se recomienda subir gráficos de "
        "<b>5 a 10 velas</b> (diarias). El modelo se entrenó con recortes chicos, "
        "centrados en el patrón, así que no está pensado para un gráfico largo con "
        "decenas de velas: en ese caso puede no detectar nada o confundirse. "
        "Lo ideal es un recorte acotado alrededor del patrón.</li>"
        "<li><b>Dominio de entrenamiento:</b> aprendió con velas sobre <b>fondo negro</b>. "
        "Para capturas con fondo blanco/grilla, dejá activado «Quitar fondo claro».</li>"
        "<li><b>Solo 6 patrones:</b> reconoce únicamente las 6 clases listadas. "
        "Cualquier otra figura o imagen fuera de ese conjunto no se detecta.</li>"
        "</ul>"
        "</div>"
    )


def pattern_explain_html(class_name: str, score: float) -> str:
    """
    HTML de la tarjeta explicativa de UN patrón detectado (sesgo + resumen).

    Se usa una vez por cada patrón distinto detectado, para explicar TODOS los
    que aparecen en la imagen y no solo el de mayor score. El contenido sale de
    PATTERN_INFO (estático); acá solo se maqueta.
    """
    info = PATTERN_INFO.get(class_name)
    if not info:
        return (
            f'<div class="card pattern-text">Se detectó el patrón '
            f"<b>{class_name}</b> (confianza {score:.0%}).</div>"
        )

    bias_cls = "up" if info["bias"] == "alcista" else "down"
    bias_dir = "al alza" if info["bias"] == "alcista" else "a la baja"
    bias_word = "alcista" if info["bias"] == "alcista" else "bajista"
    return (
        f'<div class="card explain-card">'
        f'<div class="card-title">💡 «{class_name}» '
        f'<span style="color:var(--muted);font-weight:600;font-size:.85rem">'
        f"· confianza {score:.0%}</span></div>"
        f'<div class="bias {bias_cls}">'
        f'<span class="ico">{info["icon"]}</span>'
        f"<span>Patrón de sesgo {bias_word} — la teoría lo vigila como posible "
        f"giro <b>{bias_dir} de acá en adelante</b>.</span>"
        f"</div>"
        f'<p class="pattern-text" style="margin-top:.8rem">'
        f'{md_inline_to_html(info["summary"])}</p>'
        f"</div>"
    )


# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #
def render_sidebar() -> tuple[float, bool]:
    # Encabezado propio
    st.sidebar.markdown(
        '<div class="side-title">Ajustes</div>', unsafe_allow_html=True
    )

    # Guía de uso corta (3 pasos)
    st.sidebar.markdown(
        '<div class="side-help">'
        "<b>Cómo usar</b>"
        '<ol><li>Subí una imagen del gráfico.</li>'
        "<li>Mirá el patrón y qué suele indicar.</li>"
        "<li>Si hace falta, ajustá los controles de abajo.</li></ol>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<div class="side-label">Sensibilidad</div>', unsafe_allow_html=True)
    score_threshold = st.sidebar.slider(
        "Umbral de confianza",
        min_value=0.0,
        max_value=1.0,
        value=utils.DEFAULT_SCORE_THRESHOLD,
        step=0.05,
        help="Subilo para ver solo detecciones muy seguras; bajalo para ver más.",
    )
    remove_bg = st.sidebar.checkbox(
        "Quitar fondo claro",
        value=True,
        help=(
            "El modelo se entrenó con velas sobre fondo negro. Si tu imagen tiene "
            "fondo blanco/grilla (p. ej. una captura), activá esto para quitarle el "
            "fondo antes de analizarla, como en el dataset de entrenamiento."
        ),
    )

    st.sidebar.markdown('<div class="side-label">Patrones que reconoce</div>', unsafe_allow_html=True)
    st.sidebar.markdown(class_legend_html(), unsafe_allow_html=True)

    st.sidebar.markdown(
        '<div class="side-foot">Modelo: Faster R-CNN · ResNet50-FPN</div>',
        unsafe_allow_html=True,
    )
    return score_threshold, remove_bg


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
inject_css()

# 1) Hero
st.markdown(render_hero(), unsafe_allow_html=True)

# Sidebar (controles)
score_threshold, remove_bg = render_sidebar()

# Cargar el modelo (cacheado con @st.cache_resource en utils.load_model).
model_ok = True
try:
    model, device = utils.load_model()
except Exception as exc:  # noqa: BLE001
    model_ok = False
    st.markdown(render_stats(False, "—", len(utils.CLASS_NAMES), score_threshold),
                unsafe_allow_html=True)
    st.error(f"No se pudo cargar el modelo.\n\n{exc}")
    st.stop()

dev_label = "GPU (CUDA)" if device == "cuda" else "CPU"

# 2) Fila de stats bajo el hero
st.markdown(
    render_stats(model_ok, dev_label, len(utils.CLASS_NAMES), score_threshold),
    unsafe_allow_html=True,
)

# 3) Zona de carga
st.markdown('<div class="section-title">📤 Subí tu imagen</div>', unsafe_allow_html=True)

# Contador para resetear el uploader sin recargar la página (F5). El botón
# "Subir otra imagen" incrementa el contador, lo que cambia la key del uploader
# y lo deja vacío en el siguiente rerun.
if "uploader_round" not in st.session_state:
    st.session_state["uploader_round"] = 0

uploaded = st.file_uploader(
    "Subí una imagen (PNG / JPG)",
    type=["png", "jpg", "jpeg"],
    key=f"uploader_{st.session_state['uploader_round']}",
    label_visibility="collapsed",
)

# 4) Estado vacío lindo (cuando no hay imagen)
if uploaded is None:
    st.markdown(render_empty_state(), unsafe_allow_html=True)
    st.markdown(limitations_html(), unsafe_allow_html=True)
    st.markdown(render_footer(), unsafe_allow_html=True)
    st.stop()

# Botón para limpiar y subir otra imagen sin tocar F5.
if st.button("🔄 Subir otra imagen", help="Limpia la imagen actual para analizar otra."):
    st.session_state["uploader_round"] += 1
    st.rerun()

# --- Inferencia (pipeline sin cambios) --------------------------------------
image = Image.open(BytesIO(uploaded.read()))

# Acondicionamiento OPCIONAL de la entrada (quitar fondo claro). Esto NO toca
# utils.predict ni el preprocesamiento de val/test: solo lleva la imagen subida
# al mismo dominio (fondo negro) con el que se entrenó el modelo.
if remove_bg:
    image = strip_white_background(image)

with st.spinner("Analizando la imagen…"):
    result = utils.predict(model, device, image, score_threshold=score_threshold)

detections = result["detections"]
annotated = draw_detections(result["image"], detections)

# --- Imagen anotada (ÚNICA), más grande y centrada ---------------------------
# Se muestra solo la imagen con los bounding boxes dibujados (no la original
# por separado): es la que aporta información útil sobre qué se detectó y dónde.
st.markdown('<div class="section-title">🔍 Resultado</div>', unsafe_allow_html=True)
_, col_mid, _ = st.columns([1, 2, 1])
with col_mid:
    st.markdown(
        image_card_html(annotated, "🎯 Patrón marcado", "Dónde se encontró el patrón"),
        unsafe_allow_html=True,
    )

# --- Explicación del gráfico: TODOS los patrones detectados -----------------
# Se explica cada patrón distinto encontrado (no solo el de mayor score),
# agrupando por clase y quedándose con el score más alto de cada una.
st.markdown('<div class="section-title">¿Qué muestra este gráfico?</div>', unsafe_allow_html=True)
if detections:
    # Mejor score por clase, ordenado de mayor a menor confianza.
    best_by_class: dict[str, float] = {}
    for det in detections:
        name = det["class_name"]
        if det["score"] > best_by_class.get(name, 0.0):
            best_by_class[name] = det["score"]
    patterns = sorted(best_by_class.items(), key=lambda kv: kv[1], reverse=True)

    if len(patterns) > 1:
        st.markdown(
            f'<div class="note" style="margin-bottom:.6rem">Se detectaron '
            f"<b>{len(patterns)} patrones</b> en la imagen. Se explican todos a "
            f"continuación, del más al menos confiable.</div>",
            unsafe_allow_html=True,
        )

    for class_name, score in patterns:
        st.markdown(pattern_explain_html(class_name, score), unsafe_allow_html=True)
        info = PATTERN_INFO.get(class_name)
        if info:
            with st.expander(f"Más sobre «{class_name}»"):
                st.markdown(info["detail"])

    st.markdown(
        '<div class="note">El sesgo se refiere a lo que el análisis técnico '
        "esperaría <i>después</i> del patrón, no al movimiento que ya ocurrió y "
        "forma el patrón en la imagen. Por sí solo no confirma nada y depende "
        "del contexto del gráfico.</div>",
        unsafe_allow_html=True,
    )
else:
    st.warning(
        "No se reconoció ningún patrón con claridad en esta imagen. "
        "Probá con otra captura más nítida o ajustá el umbral en la barra lateral."
    )

# Disclaimer SIEMPRE visible tras analizar una imagen (haya o no detección).
st.markdown(f'<div class="disclaimer">{DISCLAIMER_HTML}</div>', unsafe_allow_html=True)

# --- Detalle técnico (colapsado, para quien lo quiera) ----------------------
with st.expander("🔧 Ver detalle técnico"):
    if result["top"] is not None:
        st.caption(
            f"Modelo: Faster R-CNN ResNet50-FPN · "
            f"Patrón principal: {result['top']['class_name']} · "
            f"Confianza: {result['top']['score']:.1%}"
        )
    if detections:
        st.markdown("**Detecciones**")
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

    st.markdown("**Confianza por patrón**")
    st.caption("Score máximo detectado para cada uno de los 6 patrones.")
    st.markdown(class_scores_html(result["class_scores"]), unsafe_allow_html=True)

# Limitaciones siempre visibles al pie del resultado.
st.markdown(limitations_html(), unsafe_allow_html=True)

st.markdown(render_footer(), unsafe_allow_html=True)
