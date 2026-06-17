# Reconocimiento de Patrones de Velas Japonesas

Proyecto de visión por computadora con redes neuronales que **detecta 6 patrones
de velas japonesas** en imágenes de gráficos financieros, usados en análisis
técnico de mercados.

A diferencia de un clasificador simple, el modelo es un **detector de objetos**
(Faster R-CNN ResNet50-FPN): localiza cada patrón en la imagen con un *bounding
box*, su clase y un score de confianza.

## Patrones reconocidos (6 clases)

1. Bearish Engulfing
2. Bearish Insidebar
3. Bullish Engulfing
4. Bullish Insidebar
5. Hammer
6. Inverted_Hammer

## App desplegada

🔗 **[App de inferencia](https://TODO-link-de-la-app)** *(placeholder — completar con la URL del despliegue, p. ej. Streamlit Community Cloud)*

Subí una imagen de un gráfico de velas y la app detecta los patrones presentes
con su nivel de confianza. Ver [`prod/README.md`](prod/README.md) para detalles.

## Integrantes

> *Completar con los nombres del equipo:*

- Integrante 1 — *(nombre / legajo)*
- Integrante 2 — *(nombre / legajo)*
- Integrante 3 — *(nombre / legajo)*

## Estructura del repositorio

```
.
├── data/                              # CSVs de splits + README del dataset (NO modificar)
│   ├── README.md
│   ├── train.csv                      # 813 muestras (70%)
│   ├── val.csv                        # 232 muestras (20%)
│   ├── test.csv                       # 115 muestras (10%)
│   ├── raw/                           # Dataset original YOLO (no versionado)
│   └── processed/                     # Dataset con fondo removido (no versionado)
├── dev/                               # Notebooks y artefactos de desarrollo (NO modificar)
│   ├── 01_dataset_preparation.ipynb   # Preparación de datos
│   ├── 02_model_training.ipynb        # Entrenamiento (3 experimentos) y evaluación
│   ├── exp1_baseline_best.pth         # Pesos exp1 — Faster R-CNN baseline
│   ├── exp2_sampler_best.pth          # Pesos exp2 — Faster R-CNN + WeightedRandomSampler
│   ├── exp3_retinanet_best.pth        # Pesos exp3 — RetinaNet + Focal Loss
│   └── modelo.pth                     # Modelo final de producción (= exp2, ~158 MB, Git LFS)
├── prod/                              # App de inferencia (Semana 4)
│   ├── app.py                         # Interfaz Streamlit
│   ├── utils.py                       # Carga de modelo + preprocesamiento + inferencia
│   ├── requirements.txt               # Dependencias fijadas
│   └── README.md                      # Documentación de la app
├── .streamlit/
│   └── config.toml                    # Tema de la app (light, color de acento, layout)
├── dowload_dataset.py                 # Descarga del dataset desde Roboflow
├── requirements.txt                   # Dependencias del entorno de desarrollo
├── planning.md                        # Plan de implementación
└── contexto.md                        # Contexto del proyecto
```

## Modelo

- **Arquitectura:** Faster R-CNN con backbone **ResNet50-FPN** (preentrenado en
  COCO), cabeza adaptada a 6 patrones + fondo (7 clases).
- **Selección:** se compararon 3 experimentos en `dev/02_model_training.ipynb`:

  | Exp | Modelo base | Estrategia de balanceo | Val mAP@0.5 |
  |---|---|---|---|
  | 1 | Faster R-CNN | Ninguna (baseline) | ~0.74 |
  | 2 | Faster R-CNN | WeightedRandomSampler | **~0.76** ✅ |
  | 3 | RetinaNet | Focal Loss (nativa) | ~0.69 |

- **Ganador:** Experimento 2, guardado como `dev/modelo.pth`.
- **Test mAP@0.5:** ~0.77.

## Dataset

- **Fuente:** [Roboflow — Candlestick Pattern Dataset v1](https://universe.roboflow.com/madhumitha-jc-hvsdd/candlestick-pattern/dataset/1)
- **Licencia:** CC BY 4.0
- **Formato:** YOLOv8 (detección de objetos)
- **Total:** 1.160 imágenes distribuidas en 6 clases (splits 70/20/10, seed=42)

## Cómo correr el proyecto localmente

### 1. Clonar el repositorio (con Git LFS)

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

La app abre en `http://localhost:8501`. Para deploy en hostings sin Git LFS,
ver la estrategia de descarga remota en [`prod/README.md`](prod/README.md).

### 3. (Opcional) Reproducir el desarrollo

```bash
pip install -r requirements.txt
python dowload_dataset.py                       # descargar dataset
jupyter notebook dev/01_dataset_preparation.ipynb
jupyter notebook dev/02_model_training.ipynb
```

## Estado del proyecto

- [x] Descarga reproducible del dataset
- [x] Preprocesamiento (eliminación de fondo)
- [x] Splits estratificados 70/20/10 con seed fijo
- [x] Dataset y DataLoaders de PyTorch
- [x] Entrenamiento de 3 experimentos (Faster R-CNN x2, RetinaNet)
- [x] Evaluación y métricas (mAP@0.5, AP por clase)
- [x] Modelo final seleccionado y serializado (`dev/modelo.pth`)
- [x] App de inferencia (`prod/`)
- [ ] Despliegue público (completar URL)
```
