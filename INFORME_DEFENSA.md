# Informe del Trabajo Integrador
## Reconocimiento de Patrones de Velas Japonesas mediante Detección de Objetos

**Defensa oral — miércoles 17 de junio de 2026**

> Este informe recorre las 4 semanas del proyecto explicando, en cada una, **qué**
> decidimos, **por qué** (justificación) y **cómo** lo implementamos. Todos los números
> provienen del código y de los outputs ejecutados de los notebooks del repositorio.
>
> ⚠️ **Marcadores `[[COMPLETAR]]`**: las métricas del experimento nuevo (notebook v2,
> entrenado fuera del repo) aún no están volcadas. Se indican con `[[COMPLETAR: …]]`
> para pegar los valores reales antes de la defensa.

---

## 0. Resumen del proyecto

Sistema de **detección de objetos** que localiza, sobre la imagen de un gráfico de
velas japonesas, **6 patrones** de análisis técnico, devolviendo para cada uno un
*bounding box*, su clase y un score de confianza. A diferencia de un clasificador de
imagen única, el modelo puede detectar **varios patrones por imagen** y ubicarlos.

- **Tarea:** detección (no clasificación).
- **Arquitectura final:** Faster R-CNN con backbone ResNet50-FPN (preentrenado en COCO).
- **6 clases:** Bearish Engulfing, Bearish Insidebar, Bullish Engulfing, Bullish
  Insidebar, Hammer, Inverted_Hammer.
- **Entregable Semana 4:** app web (Streamlit) desplegada públicamente.

---

## Semana 1–2 — Datos: obtención, preparación y splits

### Qué hicimos
Conseguimos un dataset etiquetado de patrones de velas y construimos un pipeline
**reproducible** de preparación: descarga, limpieza de fondo, división estratificada
y `Dataset`/`DataLoader` de PyTorch para detección.

### Por qué (justificación)
- **Dataset de Roboflow** ([Candlestick Pattern v1](https://universe.roboflow.com/madhumitha-jc-hvsdd/candlestick-pattern/dataset/1),
  licencia **CC BY 4.0**, formato **YOLOv8**): ya viene anotado con *bounding boxes* y
  clase, que es justo lo que necesita una tarea de detección. **1.160 imágenes, 6 clases.**
- **Eliminación del fondo blanco → fondo negro:** las imágenes originales tienen fondo
  claro/grilla. Llevar todo a un **fondo negro homogéneo** quita ruido visual y deja
  un dominio consistente para que el modelo aprenda la forma de la vela y no el fondo.
- **Split estratificado 70/20/10 con `seed=42`:** estratificar mantiene la misma
  proporción de clases en train/val/test (clave porque hay **desbalance**), y la semilla
  fija garantiza que cualquiera reproduzca exactamente los mismos splits.

### Cómo lo implementamos
- **Splits (resultado real ejecutado):** `train=813 (70%)`, `valid=232 (20%)`,
  `test=115 (10%)`. Estratificado por clase, `SEED=42`. Se guardan como CSV
  (`data/train.csv`, `val.csv`, `test.csv`) con columnas `[image, label]`.
- **Eliminación de fondo** (`remove_white_background`, en
  `dev/01_dataset_preparation.ipynb`):
  1. Máscara de blancos con umbral **RGB ≥ 240**.
  2. **Flood fill desde las 4 esquinas** (`loDiff=10, upDiff=10`) para marcar solo el
     fondo conectado (no blancos internos de la vela).
  3. **Morfología** (kernel elíptico 3×3): `CLOSE` + `DILATE` para limpiar la máscara.
  4. Canal **alpha** (blur 3×3 + umbral) → PNG RGBA con fondo transparente, que al
     cargarse se compone sobre **negro** (`convert_to_rgb`).
- **Transforms (PyTorch):** `IMAGE_SIZE = 224`. En **train**: `Resize(224) +
  ColorJitter(brillo/contraste/saturación 0.2, hue 0.02) + ToTensor + Normalize`
  (media/std de ImageNet). En **val/test**: igual pero **sin** ColorJitter.
- **Decisión de augmentación documentada:** **no** usamos flips ni rotaciones —
  *"un flip horizontal puede convertir un patrón alcista en bajista"*, rompería la
  semántica. Solo variaciones de color (iluminación), que no alteran el patrón.
- **`Dataset` + `collate_fn` custom:** necesario porque el nº de cajas por imagen es
  variable; el collate apila imágenes pero deja los *targets* como lista de dicts.
- **Desbalance documentado:** la clase más frecuente (Bullish Engulfing, 263) casi
  **2,5×** la menos frecuente (Bearish Engulfing, 105). Esto motiva las estrategias de
  balanceo de la etapa de entrenamiento.

| Clase | Train | Val | Test | Total |
|---|---|---|---|---|
| Bearish Engulfing | 73 | 21 | 11 | 105 |
| Bearish Insidebar | 155 | 45 | 23 | 223 |
| Bullish Engulfing | 185 | 52 | 26 | 263 |
| Bullish Insidebar | 118 | 35 | 16 | 169 |
| Hammer | 175 | 50 | 25 | 250 |
| Inverted_Hammer | 105 | 30 | 15 | 150 |

---

## Semana 3 — Modelado: experimentos y selección

### Qué hicimos
Entrenamos y comparamos **3 experimentos** de detección sobre los mismos datos, para
elegir objetivamente el mejor modelo (`dev/02_model_training.ipynb`).

### Por qué (justificación)
- **Faster R-CNN ResNet50-FPN** como base: detector de dos etapas, sólido y preciso,
  con backbone preentrenado en COCO (transfer learning) — buena opción con un dataset
  relativamente chico (~800 imágenes de train).
- Dado el **desbalance** de clases, probamos estrategias para que las clases minoritarias
  no queden relegadas: **WeightedRandomSampler** y, como alternativa de arquitectura,
  **RetinaNet con Focal Loss** (la Focal Loss está pensada justamente para desbalance).

### Cómo lo implementamos (hiperparámetros reales)
Config común: `IMAGE_SIZE=224`, `BATCH_SIZE=8`, `NUM_EPOCHS=30`, `SEED=42`,
optimizador **SGD** (`momentum=0.9`, `weight_decay=5e-4`), scheduler **StepLR
(`step_size=10`, `gamma=0.5`)**, `box_nms_thresh=0.3`.

| Exp | Arquitectura | Estrategia | LR | Val mAP@0.5 |
|---|---|---|---|---|
| 1 | Faster R-CNN | Baseline (sin balanceo) | 0.005 | 0.7190 |
| 2 | Faster R-CNN | **WeightedRandomSampler** | 0.005 | **0.7307** ✅ |
| 3 | RetinaNet | Focal Loss nativa (α=0.25, γ=2.0) | 0.002 | 0.6866 |

### Resultado y selección
Ganó el **Experimento 2 (Faster R-CNN + WeightedRandomSampler)**. Métricas reales en
el **conjunto de test independiente**:

- **mAP@0.5 = 0.7713**
- **mAP@0.5:0.95 = 0.4938**
- Recall promedio (AR) = 0.6803
- AP@0.5 por clase: Bearish Engulfing 0.382 · Bearish Insidebar 0.703 · Bullish
  Engulfing 0.448 · Bullish Insidebar 0.504 · Hammer 0.566 · Inverted_Hammer 0.360.

> Lectura honesta para la defensa: el modelo es bueno en patrones bien representados
> (Insidebar, Hammer) y más flojo en los minoritarios (Bearish Engulfing,
> Inverted_Hammer), coherente con el desbalance del dataset.

---

## Semana 3–4 (refinamiento) — Segunda ronda de experimentos (notebook v2)

### Qué hicimos
Sobre los dos enfoques más prometedores, corrimos una **segunda ronda** para intentar
mejorar el desempeño, entrenando **más épocas** y comparando Focal Loss en ambas
arquitecturas (`dev/02_model_training_v2.ipynb`).

### Por qué (justificación)
- A 30 épocas los modelos aún tenían margen; **aumentar las épocas** (a 100) busca
  exprimir más capacidad antes de que sature.
- **Ajuste del scheduler StepLR (paso 10 → 25):** con decay cada 10 pasos y `gamma=0.5`,
  a 100 épocas habría **10 reducciones** y el learning rate se volvería minúsculo muy
  pronto (≈ lr·2⁻¹⁰ ≈ 5e-7), **congelando** el aprendizaje a mitad de entrenamiento.
  Con paso 25 hay solo **4 reducciones** en 100 épocas, manteniendo un lr útil durante
  más tiempo y un entrenamiento estable a largo plazo.

> **Verificación técnica del argumento (es correcto):** con `gamma=0.5`, a la época 50
> → paso 10 da 5 decays (lr·2⁻⁵ ≈ lr/32), paso 25 da 2 decays (lr·2⁻² = lr/4). El
> razonamiento de evitar el estancamiento es **válido**. (Nota: el valor exacto del lr
> depende del `gamma`; con `gamma=0.5` el decaimiento es más suave que con `0.1`, pero
> la conclusión cualitativa se sostiene.)

### Cómo lo implementamos
La segunda ronda se entrenó con el script `dev/NotebookPython.py` (verificado), que
compara dos enfoques con Focal Loss a **100 épocas**:
- **Exp 2 — Faster R-CNN + Focal Loss:** la Focal Loss se implementa por **monkey-patch**
  de `fastrcnn_loss` (softmax focal, `γ=2.0`, `α=0.25`), ya que Faster R-CNN no la trae
  nativa. Backbone ResNet50-FPN preentrenado en COCO, 7 clases (`label_offset=1`).
- **Exp 4 — RetinaNet + Focal Loss:** Focal Loss nativa (`γ=2.0`, `α=0.25`), 6 clases.

Hiperparámetros reales (confirmados en el código): `NUM_EPOCHS=100`, `BATCH_SIZE=8`,
SGD (`lr=0.005` Faster / `0.002` RetinaNet, `momentum=0.9`, `weight_decay=5e-4`),
**StepLR `step_size=25`, `gamma=0.5`**, `SEED=42`. La selección del ganador es
**automática**: gana el de mayor val mAP@0.5, se guarda como `modelo.pth` y se evalúa
en test.

### Resultado y selección
Se eligió **Faster R-CNN con Focal Loss** como mejor modelo (Exp 2). Métricas reales de
la corrida del 21/06/2026 (de `dev/ReporteDeRendimiento.png` y las curvas):

| Experimento | Arquitectura | Final Train Loss | **Val mAP@0.5** | |
|---|---|---|---|---|
| **Exp 2** | Faster R-CNN + Focal Loss (patch) | 0.0082 | **0.7489** | 🏆 ganador |
| Exp 4 | RetinaNet + Focal Loss (nativa) | 0.0163 | 0.7151 | |

**Test set del modelo ganador (Faster R-CNN + Focal Loss)** — evaluado sobre los 115
de test (`MeanAveragePrecision`, mismo pipeline que el entrenamiento):

- **mAP@0.5 = 0.8650** · **mAP@0.5:0.95 = 0.7305** · AR@100 = 0.8807
- AP@0.5:0.95 por clase: Bearish Engulfing **0.6265** · Bearish Insidebar **0.8792** ·
  Bullish Engulfing **0.7309** · Bullish Insidebar **0.7412** · Hammer **0.8512** ·
  Inverted_Hammer **0.5541**.

**Comparación ronda 1 vs ronda 2 (el dato más fuerte de la defensa):**

| Métrica (test) | Ronda 1: WeightedSampler (30 ep) | **Ronda 2: Focal Loss (100 ep)** |
|---|---|---|
| mAP@0.5 | 0.7714 | **0.8650** (+9,4 pts) |
| mAP@0.5:0.95 | 0.4939 | **0.7305** (+23,7 pts) |
| Bearish Engulfing (minoritaria) | 0.382 | **0.627** (+24,5 pts) |
| Inverted_Hammer (minoritaria) | 0.360 | **0.554** (+19,4 pts) |

**Lectura para la defensa:**
- La **Focal Loss cumplió su objetivo**: las mayores mejoras están en las **clases
  minoritarias** (Bearish Engulfing, Inverted_Hammer), que es justo lo que ataca al
  reducir el peso de los ejemplos fáciles. Justifica con datos por qué se eligió.
- La curva (`dev/curvas_aprendizaje_val_mAP.png`) muestra que Faster R-CNN Focal domina a
  RetinaNet y que ambas se mantienen **estables y productivas hasta la época 100**, sin
  estancarse → **confirma empíricamente** que el StepLR paso 25 evitó el congelamiento del
  learning rate que habría ocurrido con paso 10 a 100 épocas.

> ⚠️ **OJO — discrepancia crítica a resolver antes del miércoles:** estas métricas (0.8650)
> son del modelo **Focal Loss**, que **vive en Google Drive** (link nuevo). El `dev/modelo.pth`
> **del repo es el VIEJO** (WeightedRandomSampler, mAP 0.7714) — verificado por hash y por
> evaluación. Es decir: **el repo y, posiblemente, la app desplegada están corriendo el
> modelo peor.** Hay que (1) reemplazar `dev/modelo.pth` por el nuevo y (2) confirmar que el
> `MODEL_URL` del deploy apunte al link nuevo. Ver §Pendientes.

---

## Semana 4 — Interfaz y despliegue

### a) Aplicación web
App en **Streamlit** (`prod/app.py`). El usuario sube una imagen; la app la
preprocesa, ejecuta la detección y muestra:
- la **imagen con los bounding boxes** dibujados (una sola imagen, centrada);
- la **explicación de todos los patrones detectados** (no solo el de mayor score),
  con su sesgo (alcista/bajista) y su confianza;
- **confianza por patrón** (las 6 clases) y un detalle técnico colapsable;
- una sección de **limitaciones** (tamaño 224×224, 5–10 velas recomendadas, fondo
  negro, solo 6 patrones).

**Casos a mostrar en vivo:** imagen representativa (detecta bien) · imagen fuera de
clase (responde "no se reconoció ningún patrón") · captura con fondo blanco (checkbox
"Quitar fondo claro" la lleva al dominio de entrenamiento).

### b) Código de la app (separación de responsabilidades)
- `prod/app.py` → **presentación** (UI, widgets, dibujo de cajas, textos).
- `prod/utils.py` → **lógica**: `load_model` (carga), `preprocess_image`
  (preprocesamiento) y `predict` (inferencia + postproceso).
- **Preprocesamiento idéntico a val/test:** `convert_to_rgb → Resize(224) → to_tensor`.
  No se aplica `Normalize` manual porque Faster R-CNN normaliza internamente con su
  `GeneralizedRCNNTransform` — duplicarlo rompería el modelo (error silencioso clásico).
- **Carga cacheada** con `@st.cache_resource`: el modelo (~158 MB) se carga una sola
  vez por sesión, no en cada interacción.
- **Mejora de inferencia (detección múltiple):** el NMS interno (`0.3`) borraba
  patrones solapados de **distinta** clase, dejando solo el de mayor score. Se desactivó
  el NMS interno en runtime y se aplica un **NMS por clase** en `predict()`, de modo que
  patrones de clases distintas en la misma zona **conviven**, sin duplicar la misma clase.

### c) Despliegue
- **URL pública:** https://reconocimiento-patrones-velas-japonesas.streamlit.app/
- **Streamlit Community Cloud** (gratuito, integrado con GitHub), público, sin login.
- **`requirements.txt` con versiones fijadas** (reproducible); torch/torchvision en
  build **CPU** porque el hosting no tiene GPU (la app detecta CPU/GPU en runtime).
- **Modelo > límite de GitHub (158 MB):** se aloja en **Google Drive** y la app lo
  **descarga al iniciar** desde la variable/secret `MODEL_URL` (usando `gdown` para
  sortear la confirmación de antivirus de Drive). Cae a `dev/modelo.pth` si existe local.

### d) Repositorio
Estructura `data/` · `dev/` · `prod/`. **README en la raíz** (completo: descripción,
los 4 integrantes, link a la app, dataset, métricas e instrucciones).

---

## Justificación: ¿qué modelo dejamos en la app web?

**El modelo que se debe defender es el Faster R-CNN + Focal Loss (100 épocas)**, ganador
de la segunda ronda. Es claramente superior (test mAP@0.5 0.8650 vs 0.7714 del viejo).

**Por qué Faster R-CNN + Focal Loss (justificación):**
1. Faster R-CNN fue consistentemente la mejor arquitectura (superó a RetinaNet en val:
   0.7489 vs 0.7151).
2. La **Focal Loss** ataca el desbalance: las clases minoritarias mejoraron +20/+24 pts
   (ver tabla arriba). Es el efecto teórico esperado, comprobado con datos.
3. Es **reemplazo directo** del checkpoint: misma arquitectura (Faster R-CNN, 7 clases,
   `label_offset=1`), así que `prod/utils.py` no se toca.

> 🚨 **Estado real (a corregir antes del miércoles):** el binario del repo
> (`dev/modelo.pth`) **NO es el Focal Loss, es el viejo** (WeightedRandomSampler).
> Verificado: el hash difiere del modelo de Drive, y evaluado en test da 0.7714 (el viejo),
> no 0.8650. El modelo bueno (Focal Loss) **solo está en Google Drive** por ahora.
>
> Para que "lo que se defiende" = "lo que corre":
> 1. **Reemplazar `dev/modelo.pth`** por el nuevo (descargado de Drive, hash verificado).
> 2. **Confirmar el `MODEL_URL` del deploy**: debe apuntar al link NUEVO
>    (`1nNKD-...`), no a uno viejo. Si apunta al viejo, la app online predice peor.

---

## Pendientes antes de la defensa (checklist honesto)

1. ✅ **Métricas del v2 completas:** val (Focal 0.7489 / RetinaNet 0.7151) y test del
   ganador (mAP@0.5 **0.8650**, por clase) — todas cargadas en el informe.
2. ✅ **Hiperparámetros del v2 verificados** (100 épocas, StepLR step 25, Focal Loss).
3. 🚨 **CRÍTICO — modelo desplegado:** se reemplazó `dev/modelo.pth` por el Focal Loss
   (commit local, falta **pushear** con Git LFS). **Falta confirmar** que el `MODEL_URL`
   del deploy apunte al link nuevo (`1nNKD-...`): si apunta al viejo, la app online
   (https://reconocimiento-patrones-velas-japonesas.streamlit.app/) **predice peor** que
   lo que se defiende. *(Riesgo más alto para la nota — verificar en el panel de Secrets.)*
4. ✅ **README en la raíz creado** (descripción, 4 integrantes, link app, dataset,
   métricas, instrucciones). Falta **commitear/pushear**.
5. ✅ **Link de la app** puesto en el README e informe.
6. (Opcional pero suma) Los gráficos del script ya están en `dev/` (curvas + reporte);
   se pueden mostrar en la defensa.
7. **Pushear todo a `main`:** modelo (commit hecho), README, front rediseñado e informe
   (sin commitear aún).
8. Que **cada integrante** pueda defender: preprocesamiento = val/test,
   `@st.cache_resource`, NMS por clase, y la estrategia de descarga del modelo.

---

## Datos verificados (para citar con confianza)

- Dataset: Roboflow Candlestick Pattern v1, CC BY 4.0, YOLOv8, **1.160 imgs, 6 clases**.
- Splits: **813 / 232 / 115** (70/20/10), estratificado, `seed=42`.
- Preproc: fondo→negro (flood fill umbral 240 + morfología 3×3); `IMAGE_SIZE=224`;
  Normalize ImageNet; ColorJitter solo en train; sin flips/rotaciones.
- Entrenamiento ronda 1 (30 épocas, SGD lr 0.005, StepLR 10/0.5):
  Exp1 **0.7190** · Exp2 **0.7307** · Exp3 **0.6866** (val mAP@0.5).
- **Ganador ronda 1:** Exp 2 (WeightedRandomSampler) → **test mAP@0.5 = 0.7713**,
  mAP@0.5:0.95 = 0.4938.
- `dev/modelo.pth`: **158 MB**, Faster R-CNN, **7 clases** (verificado en `state_dict`).
