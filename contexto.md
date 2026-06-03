# Contexto del Proyecto: Reconocimiento de Patrones de Velas Japonesas

## Objetivo General

Construir un modelo de clasificación de imágenes con deep learning capaz de reconocer **6 patrones de velas japonesas** utilizados en análisis técnico de mercados financieros. El proyecto está en la etapa de preparación de datos y pipeline; el entrenamiento del modelo aún no está implementado.

---

## Estructura del Repositorio

```
Reconocimiento-Patrones-velas-Japonesas/
├── .gitignore                        # Excluye data/raw/, data/processed/, imágenes
├── .vscode/settings.json             # Configuración del entorno Python en VS Code
├── data/
│   ├── README.md                     # Documentación del dataset
│   ├── train.csv                     # 813 muestras de entrenamiento
│   ├── val.csv                       # 232 muestras de validación
│   ├── test.csv                      # 115 muestras de prueba
│   ├── raw/                          # Dataset YOLO descargado (no versionado)
│   └── processed/                    # Imágenes procesadas sin fondo (no versionadas)
├── dev/
│   └── 01_dataset_preparation.ipynb  # Notebook principal (único notebook)
├── scripts/
│   └── create_splits.py              # Script de splits estratificados
├── dowload_dataset.py                # Script de descarga del dataset
├── planning.md                       # Plan de implementación del proyecto
├── requirements.txt                  # Dependencias Python
└── README.md                         # README principal (vacío/mínimo)
```

---

## Dataset

**Fuente:** Roboflow — [Candlestick Pattern Dataset v1](https://universe.roboflow.com/madhumitha-jc-hvsdd/candlestick-pattern/dataset/1)  
**Licencia:** CC BY 4.0  
**Formato original:** YOLOv8 (adaptado para clasificación)

### Clases (6 total)

| ID | Clase |
|----|-------|
| 0 | Bearish Engulfing |
| 1 | Bearish Insidebar |
| 2 | Bullish Engulfing |
| 3 | Bullish Insidebar |
| 4 | Hammer |
| 5 | Inverted Hammer |

### Distribución de Datos

| Split | Muestras | Porcentaje |
|-------|----------|------------|
| Train | 813 | 70% |
| Val | 232 | 15% |
| Test | 115 | 15% |
| **Total** | **1,160** | 100% |

- Splits estratificados con `seed=42` para reproducibilidad.
- Balanceados por clase en cada split.
- Las rutas relativas a imágenes y etiquetas están almacenadas en los CSV.

---

## Pipeline de Preparación de Datos

### 1. Descarga (`dowload_dataset.py`)
- Descarga el dataset desde Roboflow usando su API (API key hardcodeada — riesgo de seguridad).
- Coloca el dataset en `data/raw/`.

### 2. Preprocesamiento y Eliminación de Fondo (`dev/01_dataset_preparation.ipynb`, Celda 1)
- Elimina el fondo blanco de cada imagen (umbral=240) usando flood fill de OpenCV.
- Aplica operaciones morfológicas (MORPH_CLOSE, MORPH_DILATE).
- Genera imágenes RGBA con canal alpha.
- Salida: `data/processed/dataset_no_background/`
- Velocidad medida: ~8.88 img/s (aprox. 1 min 31 s en el set de entrenamiento).

### 3. Creación de Splits (`scripts/create_splits.py`)
- Lee archivos de etiquetas YOLO (.txt) para extraer el class ID.
- Genera splits estratificados (70/15/15).
- Guarda rutas relativas en `data/train.csv`, `data/val.csv`, `data/test.csv`.

---

## Implementación PyTorch

### Dataset Personalizado (`CandlestickDataset`)
- Hereda de `torch.utils.data.Dataset`.
- Carga imágenes desde los CSV y aplica transformaciones.
- Convierte RGBA a RGB si es necesario.

### Transformaciones

**Entrenamiento:**
```
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)
→ Resize(224, 224)
→ ToTensor()
→ Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

**Validación / Test:**
```
Resize(224, 224)
→ ToTensor()
→ Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

> La normalización usa valores de ImageNet, lo que indica que se planea **transfer learning** con un backbone preentrenado en ImageNet.

### DataLoaders

| Parámetro | Valor |
|-----------|-------|
| Batch size | 16 |
| num_workers | 2 |
| shuffle (train) | True |
| shuffle (val/test) | False |
| Forma del batch | (16, 3, 224, 224) |

**Problema conocido:** Error de pickling con funciones lambda cuando `num_workers > 0`. Solución: usar `num_workers=0` o reemplazar lambdas por funciones nombradas.

---

## Dependencias (`requirements.txt`)

| Librería | Uso |
|----------|-----|
| `roboflow` | Descarga del dataset |
| `pandas` | Manejo de CSV |
| `numpy` | Cómputo numérico |
| `matplotlib` | Visualización |
| `jupyter` | Entorno de notebooks |
| `pyyaml` | Parseo de data.yaml |
| `torch` | Framework de deep learning |
| `torchvision` | Transforms, DataLoader |
| `Pillow` | Carga de imágenes |
| `opencv-python` | Eliminación de fondo (cv2) |
| `tqdm` | Barras de progreso |

---

## Cambios Recientes — Commit `7243cb0` (03/06/2026, MVisedo)

**"fix: codigo de batch y grafica de boxes"**

### Cambios funcionales

1. **`NUM_WORKERS` corregido: `2` → `0`**  
   Fix definitivo para el error de pickling con lambdas en Windows/Jupyter. Se agrega comentario explicativo.

2. **Función nombrada `convert_to_rgb`** en lugar de lambda  
   Reemplaza el `T.Lambda(lambda img: ...)` en los transforms para que el DataLoader pueda serializarla con `num_workers > 0` en el futuro.

3. **Ratios de split ajustados: 70/15/15 → 70/20/10**  
   `VAL_RATIO = 0.2`, `TEST_RATIO = 0.1`. Los números finales (813/232/115) quedan iguales porque el dataset es el mismo y el split ya estaba generado.

4. **Flag `REGENERATE_SPLITS = True`** agregado  
   Fuerza la regeneración de splits aunque ya existan las carpetas.

5. **Lógica de splits integrada al notebook**  
   La función `create_stratified_splits()` ahora vive dentro del notebook (antes estaba solo en `scripts/create_splits.py`). Incluye manejo de backup para regeneración segura.

6. **Nueva función `read_yolo_annotations()`**  
   Lee **todos** los bounding boxes de un archivo de etiqueta YOLO (devuelve lista de tuplas `(class_id, x_center, y_center, width, height)`). Antes solo se leía el class ID del primer box.

7. **Nueva función `read_yolo_class()`**  
   Wrapper que usa `read_yolo_annotations()` para obtener solo la clase del primer box (mantiene compatibilidad con `__getitem__`).

8. **Nuevo método `get_raw_sample(idx)`** en `CandlestickDataset`  
   Retorna imagen PIL sin transformar + lista de boxes YOLO. Diseñado para visualización.

9. **Nueva función `draw_yolo_boxes()`**  
   Dibuja bounding boxes sobre imágenes PIL con colores por box y etiqueta de clase.

10. **Visualizaciones renovadas:**  
    - Se muestran imágenes en 2 columnas: original + con boxes dibujados  
    - El batch visualizado muestra imágenes desnormalizadas con boxes superpuestos  
    - Se visualizan 8 ejemplos del dataset train con todos sus boxes

11. **Velocidad de procesamiento mejorada:** 8.88 img/s → 46.94 img/s (diferente máquina: usuario `matia`)

---

## Estado Actual del Proyecto

### Completado
- Infraestructura de descarga del dataset
- Eliminación de fondo (preprocessing)
- Generación de splits estratificados y CSV (con flag de regeneración)
- Clase `CandlestickDataset` con soporte de múltiples boxes por imagen
- Configuración de DataLoaders (NUM_WORKERS=0 para compatibilidad Windows)
- Pipeline de data augmentation (ColorJitter solo en train)
- Visualización de bounding boxes YOLO sobre imágenes
- Lectura completa de anotaciones YOLO (múltiples boxes por imagen)

### Pendiente de Implementar
- Selección y definición de la arquitectura del modelo (transfer learning)
- Loop de entrenamiento
- Métricas de evaluación (accuracy, F1, matriz de confusión)
- Checkpointing del modelo
- Ajuste de hiperparámetros
- Reporte de resultados

---

## Observaciones Importantes

1. **Transfer Learning esperado:** La normalización ImageNet y el tamaño 224x224 apuntan a usar un backbone como ResNet, EfficientNet o similar.
2. **Reproducibilidad:** `seed=42` usado consistentemente en todo el proyecto.
3. **Seguridad:** El script `dowload_dataset.py` tiene una API key de Roboflow hardcodeada — no debería subirse al repositorio público.
4. **Idioma:** La documentación y comentarios del proyecto están principalmente en español.
5. **Una sola notebook:** Todo el trabajo de preparación de datos está en `dev/01_dataset_preparation.ipynb`.
6. **Dataset relativamente pequeño:** 1,160 imágenes totales — el data augmentation y el transfer learning son cruciales para un buen desempeño.
7. **Múltiples boxes por imagen:** El dataset puede tener más de un bounding box por imagen; el modelo de clasificación usa solo el box principal (primer box), pero la infraestructura ya lee todos los boxes para visualización.
8. **Dos desarrolladores activos:** `emili` (EmilianoJordan11) y `matia` (MVisedo) trabajan en el proyecto desde máquinas distintas.

---

## Cambios en rama `vichen` (03/06/2026) — completar requisitos de la Segunda Semana

Cambios hechos para cumplir 100% con el enunciado de la consigna:

### Archivos nuevos
- **`prod/`** — carpeta placeholder con README explicando que aloja código de inferencia futuro.
- **`README.md`** (raíz) — descripción del proyecto, estructura, cómo correrlo (estaba vacío).
- **`scripts/_build_notebook.py`** — script que regenera el notebook de forma determinística (no es parte del pipeline).

### Notebook `dev/01_dataset_preparation.ipynb` — rehecho completamente
- **Reestructurado a 23 celdas** (12 markdown + 11 código) con secciones numeradas y narrativa clara para presentación oral.
- **Comentarios extensivos en TODO el código** — cada bloque y función explica qué hace y por qué.
- **NUEVO: Sección 5 — Distribución de clases por split**
  - Tabla con conteos por clase × split (train / valid / test / total).
  - Gráfico de barras agrupadas con anotaciones numéricas.
  - Tabla de proporciones (%) por clase dentro de cada split — verifica que la estratificación funcionó.
- **NUEVO: Sección 7 — Visualización real del efecto de ColorJitter**
  - Grilla 4 imágenes × (1 original + 3 versiones aumentadas) = 12 paneles.
  - Justificación de por qué se usa ColorJitter y NO flips/rotaciones (preservar semántica del patrón).
- **Sección 10 — Bounding boxes YOLO** marcada como exploración opcional.

### Scripts comentados
- **`dowload_dataset.py`** — docstring + comentarios por bloque.
- **`scripts/create_splits.py`** — docstrings en cada función + comentarios línea a línea del algoritmo de estratificación.

### Cumplimiento del enunciado tras los cambios

| Requisito | Estado |
|---|---|
| Carga y organización (Dataset, DataLoaders) | OK |
| Particionado + distribución por clase + seed | OK |
| Preprocesamiento (resize 224, ImageNet norm) | OK |
| Data augmentation + **visualización del efecto** | OK |
| Verificación final del batch | OK |
| Repo con `data/`, `dev/`, `prod/` | OK |
| `.gitignore` excluye imágenes | OK |
| README de raíz + README de data/ | OK |
| CSV de splits versionados | OK |
| Reproducibilidad punta a punta | OK |
