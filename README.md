# Reconocimiento de Patrones de Velas Japonesas

Proyecto de visión por computadora que **detecta 6 patrones de velas japonesas** en
imágenes de gráficos financieros, usados en análisis técnico de mercados.

A diferencia de un clasificador simple, el modelo es un **detector de objetos**
(Faster R-CNN ResNet50-FPN): localiza cada patrón con un *bounding box*, su clase y un
score de confianza, pudiendo detectar **varios patrones por imagen**.

## 🔗 App desplegada

**[Abrir la app de inferencia](https://reconocimiento-patrones-velas-japonesas.streamlit.app/)**

Subí una imagen de un gráfico de velas y la app detecta los patrones presentes, los
marca con su caja y explica qué significa cada uno.

## 👥 Integrantes

- Dallapé, Vincenzo
- Díaz, Octavio
- Jordan, Emiliano
- Visedo, Matías

## Patrones reconocidos (6 clases)

1. Bearish Engulfing
2. Bearish Insidebar
3. Bullish Engulfing
4. Bullish Insidebar
5. Hammer
6. Inverted_Hammer

## Modelo

- **Arquitectura:** Faster R-CNN con backbone **ResNet50-FPN** (preentrenado en COCO),
  cabeza adaptada a 6 patrones + fondo (7 clases).
- **Modelo final:** Faster R-CNN entrenado con **Focal Loss** (100 épocas, StepLR con
  `step_size=25`), que ataca el desbalance de clases del dataset.
- **Desempeño en test (115 imágenes):**

  | Métrica | Valor |
  |---|---|
  | mAP@0.5 | **0.8650** |
  | mAP@0.5:0.95 | 0.7305 |
  | AR@100 | 0.8807 |

  AP@0.5:0.95 por clase: Bearish Engulfing 0.627 · Bearish Insidebar 0.879 · Bullish
  Engulfing 0.731 · Bullish Insidebar 0.741 · Hammer 0.851 · Inverted_Hammer 0.554.

- **Selección:** se comparó contra Faster R-CNN baseline, WeightedRandomSampler y
  RetinaNet (con y sin Focal Loss). El detalle está en `dev/` y en el informe del proyecto.

## Dataset

- **Fuente:** [Roboflow — Candlestick Pattern v1](https://universe.roboflow.com/madhumitha-jc-hvsdd/candlestick-pattern/dataset/1)
- **Licencia:** CC BY 4.0 · **Formato:** YOLOv8 (detección de objetos)
- **Total:** 1.160 imágenes, 6 clases · splits **70/20/10** estratificados (`seed=42`):
  train 813 · val 232 · test 115.

## Estructura del repositorio

```
.
├── data/                          # Splits (CSV) del dataset
│   ├── train.csv / val.csv / test.csv
│   ├── raw/                       # Dataset original YOLO (no versionado)
│   └── processed/                 # Dataset con fondo removido (no versionado)
├── dev/                           # Desarrollo: notebooks, script y artefactos
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_model_training.ipynb    # Primera ronda (3 experimentos)
│   ├── 02_model_training_v2.ipynb # Segunda ronda
│   ├── NotebookPython.py          # Script de la 2ª ronda (Focal Loss, 100 épocas)
│   ├── *.png                      # Curvas y reporte de rendimiento
│   └── modelo.pth                 # Modelo final (~158 MB, Git LFS)
├── prod/                          # App de inferencia (Semana 4)
│   ├── app.py                     # Interfaz Streamlit
│   ├── utils.py                   # Carga del modelo + preprocesamiento + inferencia
│   ├── requirements.txt           # Dependencias fijadas
│   └── assets/                    # Logo
├── .streamlit/                    # Config y ejemplo de secrets
└── README.md
```

## Cómo correr el proyecto localmente

### 1. Clonar (con Git LFS)

El modelo `dev/modelo.pth` (~158 MB) se versiona con **Git LFS**.

```bash
git lfs install
git clone https://github.com/EmilianoJordan11/Reconocimiento-Patrones-velas-Japonesas.git
cd Reconocimiento-Patrones-velas-Japonesas
git lfs pull          # descarga el binario real del modelo
```

### 2. Ejecutar la app de inferencia

```bash
pip install -r prod/requirements.txt
streamlit run prod/app.py
```

La app abre en `http://localhost:8501`. Para el despliegue (hosting sin Git LFS), el
modelo se descarga al iniciar desde la variable/secret `MODEL_URL` — ver
[`prod/README.md`](prod/README.md).

### 3. (Opcional) Reproducir el desarrollo

```bash
pip install -r requirements.txt
python download_dataset.py
jupyter notebook dev/01_dataset_preparation.ipynb
jupyter notebook dev/02_model_training.ipynb
```

## Preprocesamiento (idéntico en entrenamiento y producción)

`convert_to_rgb` (compone sobre fondo negro) → `Resize(224×224)` → `to_tensor`. **No**
se aplica `Normalize` manual en producción: Faster R-CNN normaliza internamente con su
`GeneralizedRCNNTransform`. Mantener este pipeline idéntico a val/test es clave para que
el modelo funcione correctamente en la app.
