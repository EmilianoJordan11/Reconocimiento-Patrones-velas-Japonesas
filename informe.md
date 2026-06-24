# Informe Técnico — Reconocimiento de Patrones de Velas Japonesas

**Materia:** Redes Neuronales Profundas — Ingeniería en Sistemas de Información
**Trabajo Práctico Integrador 2026**

**Integrantes:** Dallape Vicenzo · Diaz Octavio · Jordan Emiliano · Visedo Matias

**App desplegada:** <https://reconocimiento-patrones-velas-japonesas.streamlit.app/>
**Dataset:** [Roboflow — Candlestick Pattern v1](https://universe.roboflow.com/madhumitha-jc-hvsdd/candlestick-pattern/dataset/1)
**Repositorio:** estructura `data/` · `dev/` · `prod/`

> **En una línea:** una aplicación web que recibe una imagen de un gráfico de
> velas japonesas y **detecta y localiza** (con *bounding boxes*) cuál de 6
> patrones de análisis técnico aparece, indicando además si su sesgo histórico es
> alcista o bajista.

---

## Índice

1. [Problema y dataset](#1-problema-y-dataset)
2. [Preparación de datos](#2-preparación-de-datos)
3. [Entrenamiento](#3-entrenamiento)
4. [Aplicación y despliegue](#4-aplicación-y-despliegue)
5. [Cierre: decisiones y justificaciones](#5-cierre-resumen-de-decisiones-y-justificaciones)

---

## 1. Problema y dataset

### 1.1. Problema concreto

El sistema resuelve un problema de **detección y localización multiclase (Object
Detection)** de patrones de velas japonesas (*candlestick patterns*) en imágenes
de gráficos financieros, con el fin de asistir la lectura de tendencias de
mercado. No es un clasificador de imagen única: por cada gráfico el modelo puede
detectar **varios patrones**, devolviendo para cada uno su **caja
delimitadora**, su **clase** y un **score de confianza**.

Se reconocen **6 patrones (clases)**:

| # | Patrón | Sesgo histórico |
|---|--------|-----------------|
| 1 | Bearish Engulfing (envolvente bajista) | Bajista |
| 2 | Bearish Insidebar (barra interior bajista) | Bajista |
| 3 | Bullish Engulfing (envolvente alcista) | Alcista |
| 4 | Bullish Insidebar (barra interior alcista) | Alcista |
| 5 | Hammer (martillo) | Alcista |
| 6 | Inverted Hammer (martillo invertido) | Alcista |

### 1.2. Interacción en la web app

El usuario objetivo (trader o analista financiero) **sube una captura de pantalla
del gráfico de cotización** de cualquier activo (criptomonedas, acciones o Forex).
La aplicación procesa el gráfico visualmente y devuelve:

- la imagen de entrada con **cajas de selección** que encierran e identifican los
  patrones detectados (p. ej., dónde hay un *Hammer* o un *Engulfing*),
- un panel que indica si la señal es **alcista o bajista**, con su nivel de
  confianza y una explicación educativa del patrón.

### 1.3. Justificación: ¿por qué una red neuronal?

Identificar patrones de velas requiere evaluar **contexto visual**: la proporción
matemática exacta entre el cuerpo de la vela, la mecha superior y la mecha
inferior, en relación con las velas anteriores. Programar esto con **reglas
tradicionales de código es extremadamente rígido** y falla apenas el gráfico
cambia de escala, resolución, paleta de colores o incorpora indicadores de fondo
(medias móviles, volumen, grillas). Una regla codificada a mano sobre píxeles o
sobre umbrales fijos no generaliza a esa variabilidad.

El **fine-tuning de un modelo preentrenado de detección** (variantes de Faster
R-CNN / RetinaNet, comparadas más abajo contra YOLOv8) permite **extraer las
características morfológicas abstractas** de las figuras —cuerpo, mechas, relación
con la vela previa— **sin importar el formato visual** del gráfico. La red aprende
la *forma* del patrón, no una receta de píxeles, y por eso tolera cambios de
escala, color y fondo que romperían una solución basada en reglas.

### 1.4. Análisis del dataset

- **Fuente:** Roboflow — *Candlestick Pattern*, workspace `madhumitha-jc-hvsdd`,
  versión 1. Descarga automatizada y reproducible vía `download_dataset.py`.
- **Licencia:** CC BY 4.0.
- **Formato:** YOLOv8 (detección de objetos): cada imagen viene con un `.txt` de
  anotaciones, una línea por caja `clase x_center y_center width height`
  (coordenadas normalizadas en `[0, 1]`).
- **Volumen:** **1.160 imágenes** con **1.160 cajas** anotadas, repartidas en las
  6 clases.
- **Particionado (ver §2):** splits estratificados **70 / 20 / 10** con
  `seed = 42`.

| Split | Imágenes | Porcentaje |
|-------|----------|------------|
| Train | 813 | 70 % |
| Val   | 232 | 20 % |
| Test  | 115 | 10 % |

**Distribución de cajas por clase** (calculada directamente de los archivos de
etiquetas):

| Clase | Train | Val | Test | Total |
|-------|------:|----:|-----:|------:|
| Bearish Engulfing | 73 | 21 | 11 | **105** |
| Bearish Insidebar | 155 | 45 | 23 | **223** |
| Bullish Engulfing | 185 | 52 | 26 | **263** |
| Bullish Insidebar | 118 | 35 | 16 | **169** |
| Hammer | 175 | 50 | 25 | **250** |
| Inverted Hammer | 105 | 30 | 15 | **150** |
| **Total** | **813** | **232** | **115** | **1.160** |

**Observación clave — desbalance de clases:** existe un desbalance real de
aproximadamente **2,5×** entre la clase más frecuente (*Bullish Engulfing*, 263) y
la menos frecuente (*Bearish Engulfing*, 105). Este desbalance es **moderado pero
suficiente** para sesgar el aprendizaje hacia las clases mayoritarias, y es la
motivación directa de las técnicas de balanceo evaluadas en el entrenamiento
(WeightedRandomSampler y Focal Loss; ver §3.3). La estratificación garantiza que
esa proporción se conserve idéntica en train, val y test, para que la evaluación
sea representativa.

> **Condiciones de captura:** las imágenes son gráficos de velas renderizados.
> Originalmente vienen sobre fondo claro/blanco; en §2 se documenta por qué y cómo
> se las lleva a fondo negro antes de entrenar.

---

## 2. Preparación de datos

Todo el pipeline de preparación está en
[`dev/01_dataset_preparation.ipynb`](dev/01_dataset_preparation.ipynb) y es
**reproducible de punta a punta** (clonar → instalar dependencias → descargar
dataset → correr notebook, sin pasos manuales intermedios). Las imágenes **no se
versionan** (`.gitignore`); en el repo viven solo los CSV de splits
(`data/train.csv`, `data/val.csv`, `data/test.csv`), que son livianos y aseguran
que todos trabajen con el mismo particionado.

### 2.1. Remoción del fondo blanco (decisión de dominio)

**Qué se hace:** se transforma cada imagen quitando el fondo claro y dejando las
velas sobre **fondo negro** (se guarda RGBA, con el canal alpha como máscara:
0 = fondo, 255 = vela). El algoritmo es un **flood-fill desde las cuatro
esquinas** (que en estos gráficos son siempre fondo), con umbral de blanco 240,
seguido de operaciones morfológicas (*close* + *dilate*) para suavizar los bordes.

**Por qué:** unifica el **dominio visual** de todas las imágenes y elimina ruido
de fondo (grillas, color de tema) que no aporta a la morfología del patrón. Esta
misma operación se replica **idéntica** en la app de producción
(`prod/app.py::remove_white_background`), para que las capturas reales del usuario
—que suelen tener fondo blanco— se lleven al **mismo dominio** con el que se
entrenó. Es una de las decisiones que más impacta la coherencia
entrenamiento ↔ inferencia.

### 2.2. Splits estratificados y reproducibles

- **Criterio:** partición **estratificada por clase** 70/20/10. Se agrupan las
  muestras por su clase y se reparte cada grupo en las mismas proporciones, de
  modo que **ninguna clase quede sub-representada** en val o test (crítico con un
  dataset chico y desbalanceado).
- **Reproducibilidad:** `seed = 42` fijada antes de barajar → la partición es
  **idéntica en cualquier máquina**. El resultado se persiste en
  `data/{train,val,test}.csv` con las rutas relativas de cada imagen y su label.

### 2.3. `Dataset` de PyTorch para detección

Se implementó una clase propia que **hereda de `torch.utils.data.Dataset`**
(`CandlestickDataset` / `CandlestickDetectionDataset`). Su `__getitem__` devuelve
`(imagen, target)`, donde `target` es un diccionario con:

- `boxes`: tensor `[N, 4]` con las coordenadas de las cajas,
- `labels`: tensor `[N]` con la clase de cada caja.

Como **N (cajas por imagen) varía** entre muestras, un `DataLoader` estándar
fallaría al intentar apilar los targets. Por eso se usa un **`collate_fn`
personalizado** (`detection_collate_fn`) que apila las imágenes en un tensor
`[B, C, H, W]` pero deja los targets como **lista de diccionarios** sin apilar.

En el notebook de entrenamiento, el dataset agrega un parámetro **`label_offset`**
para alternar el índice de clases según la arquitectura:

- **Faster R-CNN** → `label_offset = 1` (las clases van de 1 a 6; el índice 0
  queda **reservado para el fondo**).
- **RetinaNet** → `label_offset = 0` (indexa las clases directamente de 0 a 5).

Además se transforman las coordenadas: el dataset de detección de torchvision
espera formato **Pascal VOC absoluto `[x1, y1, x2, y2]`**, por lo que se convierte
desde el formato YOLO normalizado `(x_center, y_center, w, h)` con `yolo_to_xyxy`,
escalado a 224×224.

### 2.4. Preprocesamiento

- **Resize a 224×224**: tamaño esperado por los backbones tipo ResNet/FPN
  preentrenados.
- **`convert_to_rgb`**: las imágenes son RGBA (por la remoción de fondo); se las
  compone **sobre fondo negro** usando el alpha como máscara. Usar `convert("RGB")`
  directo descartaría el alpha y volvería a mostrar el fondo blanco original — de
  ahí la función específica.
- **Normalización ImageNet** (`mean = [0.485, 0.456, 0.406]`,
  `std = [0.229, 0.224, 0.225]`) en el pipeline del notebook de preparación, más
  conversión a tensor.

> **Nota de coherencia (importante):** en la app de producción **no** se aplica un
> `Normalize` externo, porque Faster R-CNN **normaliza internamente** con su propio
> `GeneralizedRCNNTransform` (media/desvío de COCO). Agregar un `Normalize` manual
> en inferencia rompería silenciosamente el modelo. Esta decisión se detalla en §4.

### 2.5. Data augmentation (solo en train)

Se aplica **únicamente al conjunto de entrenamiento**:

```
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)
```

**Por qué ColorJitter:** simula variaciones de iluminación y de paleta de colores
entre plataformas de trading (temas claros/oscuros, distintos colores de velas
alcistas/bajistas). Hace al modelo robusto al *aspecto* del gráfico sin alterar la
morfología del patrón.

**Por qué NO flips ni rotaciones:** son augmentations habituales en visión, pero
acá **romperían la semántica del problema**. Un **flip horizontal o vertical puede
convertir un patrón alcista en uno bajista** (por ejemplo, invertir un *Hammer*),
generando pares imagen–etiqueta incorrectos. La dirección y la posición de las
mechas son parte de la definición del patrón, así que esas transformaciones se
descartan deliberadamente.

Los **bounding boxes se conservan** bajo ColorJitter (sólo cambia el color, no la
geometría), y en el notebook se **visualiza el efecto** sobre varias imágenes para
confirmarlo.

### 2.6. DataLoaders y verificación final

- `DataLoader` para train / val / test con el `collate_fn` de detección;
  `shuffle = True` solo en train.
- **Verificación:** se toma un batch de train, se **desnormalizan** las imágenes y
  se **dibujan sus cajas** con la etiqueta correspondiente, confirmando
  visualmente que los pares imagen–target están bien alineados. Se reportan
  dimensiones de los tensores `[B, C, H, W]` y el rango de valores tras la
  normalización.

---

## 3. Entrenamiento

> **Estado:** etapa **cerrada**. La ronda final entrenó a **100 épocas** los dos
> mejores candidatos —**Exp2: Faster R-CNN + Focal Loss** y **Exp4: RetinaNet +
> Focal Loss**— y de su comparación salió el **modelo ganador: Faster R-CNN +
> Focal Loss** (val mAP@0.5 0.7489 vs 0.7151; **test mAP@0.5 0.8650**). Ese modelo
> es el que sirve la app. Las métricas finales están en §3.5.

El desarrollo experimental vive en
[`dev/02_model_training.ipynb`](dev/02_model_training.ipynb) (v1),
[`dev/02_model_training_v2.ipynb`](dev/02_model_training_v2.ipynb) (v2) y el script
[`dev/NotebookPython.py`](dev/NotebookPython.py) (ronda final a 100 épocas, ejecutable
en terminal con multiprocesamiento). Las curvas y el reporte de esa ronda están en
`dev/` (`curvas_aprendizaje_*.png`, `ReporteDeRendimiento.png`).

### 3.1. Modelos preentrenados y sus características

Ambos candidatos parten de un **backbone ResNet50 con FPN (Feature Pyramid
Network)** preentrenado en **COCO** (`weights="COCO_V1"`). El FPN construye una
**pirámide de características multiescala**, lo que ayuda a detectar tanto patrones
grandes (un *engulfing* que abarca varias velas) como detalles finos (mechas
delgadas de un *hammer*).

**Candidato A — Faster R-CNN ResNet50-FPN (detector de dos etapas):**

- **Etapa 1 — RPN (Region Proposal Network):** propone regiones candidatas
  (*anchors* refinados) donde podría haber un objeto.
- **Etapa 2 — ROI heads:** para cada propuesta, clasifica el patrón y **refina la
  caja** (regresión). La cabeza de clasificación/regresión (`box_predictor`) se
  **reemplaza** por un `FastRCNNPredictor` con `num_classes = 7` (6 patrones +
  fondo).
- Se fija `box_nms_thresh = 0.3` para un *Non-Max Suppression* más estricto
  (menos cajas redundantes solapadas).
- **Característica distintiva:** la doble etapa lo hace **más preciso en la
  localización**, especialmente con objetos pequeños o solapados, a costa de mayor
  cómputo y tamaño (≈158 MB de pesos).

**Candidato B — RetinaNet ResNet50-FPN (detector de una etapa):**

- Evalúa directamente **miles de anchors** sobre la pirámide de características, sin
  una etapa previa de propuestas: un único paso de detección (como YOLO).
- **Focal Loss nativa** en su cabeza de clasificación (`RetinaNetClassificationHead`,
  reemplazada para `num_classes = 6`). La Focal Loss fue diseñada precisamente para
  el **enorme desbalance** entre los pocos anchors positivos y la mayoría de fondo
  de un detector denso.
- **Característica distintiva:** **una sola etapa**, por lo tanto más liviano y
  rápido, con el desbalance tratado *by design* a nivel de función de pérdida.

### 3.2. Estrategia de fine-tuning y justificación

Para ambos modelos se aplicó **transfer learning desde COCO + full fine-tuning**:

1. **Se parte de pesos preentrenados en COCO** (no desde cero): el backbone ya sabe
   extraer bordes, texturas y formas genéricas, lo que con un dataset chico
   (1.160 imágenes) es la diferencia entre converger o no.
2. **Se reemplaza únicamente la cabeza** (`box_predictor` en Faster R-CNN,
   `classification_head` en RetinaNet) para adaptarla a **nuestro número de
   clases**. COCO tiene 80 clases; nuestro problema, 6 (+ fondo en el caso de las
   dos etapas).
3. **Full fine-tuning (no se congela ninguna capa):** todos los parámetros quedan
   con `requires_grad = True`.

   **Por qué entrenar todo y no congelar el backbone:** el dominio de destino
   —gráficos sintéticos de velas sobre fondo negro— es **muy distinto** al dominio
   de COCO (fotos naturales de objetos cotidianos). Las características de bajo y
   medio nivel útiles para una foto no son óptimas para distinguir la proporción
   cuerpo/mecha de una vela. Congelar el backbone obligaría a clasificar con
   *features* del dominio equivocado. Como el dataset, aunque chico, es **suficiente
   y muy homogéneo** (todas las imágenes son del mismo tipo), conviene **readaptar
   toda la red** para que aprenda la morfología específica del problema. El riesgo
   de sobreajuste se mitiga con la augmentation de §2.5, el `weight_decay` y la
   selección del mejor checkpoint por mAP de validación.

4. **Manejo del índice de fondo (`label_offset`):** Faster R-CNN reserva el índice
   `0` para "fondo", por eso sus clases van de 1 a 6 (`label_offset = 1`).
   RetinaNet no lo reserva y usa 0–5 (`label_offset = 0`). En inferencia se vuelve
   a mapear `label − 1` para mostrar el nombre del patrón.

**Configuración de entrenamiento (común a los experimentos):**

| Hiperparámetro | Faster R-CNN | RetinaNet |
|---|---|---|
| Optimizador | SGD (momentum 0.9, weight_decay 5e-4) | SGD (momentum 0.9, weight_decay 5e-4) |
| Learning rate inicial | 0.005 | 0.002 |
| Scheduler | StepLR (step=10, gamma=0.5) | StepLR (step=10, gamma=0.5) |
| Batch size | 8 | 8 |
| Épocas (v3) | 100 | 100 |
| Métrica de selección | mAP@0.5 (validación) | mAP@0.5 (validación) |

- **Función de pérdida:** las pérdidas nativas de detección de torchvision
  (clasificación + regresión de cajas). En los experimentos "Focal" se sustituye la
  componente de **clasificación** por **Focal Loss** (γ = 2.0, α = 0.25) para
  atacar el desbalance: en Faster R-CNN vía un *monkey-patch* puntual de
  `fastrcnn_loss` con una `softmax_focal_loss`; en RetinaNet activando su Focal
  nativa.
- **Métrica:** **mAP@0.5** y **mAP@0.5:0.95** (mean Average Precision) con
  `torchmetrics`, además de **AP por clase** para el reporte final. El loop guarda
  el **mejor checkpoint según mAP@0.5 de validación**.
- **Hardware:** GPU local (build CUDA cu128) para desarrollo; la app de producción
  detecta CPU/GPU automáticamente en runtime.

### 3.3. Camino experimental (al menos 3 configuraciones)

La experimentación recorrió, de forma incremental, el tratamiento del desbalance:

| Iteración | Experimento | Modelo | Estrategia de balanceo | Qué se buscaba aprender |
|---|---|---|---|---|
| v1 | Exp 1 — Baseline | Faster R-CNN | Ninguna | Comportamiento base frente a clases minoritarias |
| v1 | Exp 2 — Sampler | Faster R-CNN | WeightedRandomSampler (sobremuestreo) | Si forzar paridad de clases en el batch mejora las clases raras |
| v1 | Exp 3 — RetinaNet | RetinaNet | Focal Loss nativa | Si una arquitectura 1-etapa pensada para desbalance rinde mejor |
| v2 | Exp 1 — Baseline | Faster R-CNN | Ninguna | Re-baseline con 40 épocas |
| v2 | Exp 2 — **FasterRCNN Focal** | Faster R-CNN | Focal Loss (patch en `fastrcnn_loss`) | Llevar la Focal Loss a la 2-etapas |
| v2 | Exp 3 — RetinaNet Baseline | RetinaNet | Ninguna (γ=0) | Aislar el efecto de la Focal en RetinaNet |
| v2 | Exp 4 — **RetinaNet Focal** | RetinaNet | Focal Loss (γ=2.0) | Confirmar el aporte de la Focal en 1-etapa |
| **Final** | **Exp 2 + Exp 4** | FRCNN / RetinaNet | Focal Loss | **Ganador definido a 100 épocas: FRCNN Focal** |

**Lectura del recorrido:** se empezó con un baseline para medir el punto de
partida; se probó **balanceo por datos** (WeightedRandomSampler, que sobremuestrea
las clases raras en el batch) y luego **balanceo por la función de pérdida**
(Focal Loss, que baja el peso de los ejemplos fáciles/mayoritarios y enfoca el
aprendizaje en los difíciles). La Focal Loss resultó la línea más prometedora en
ambas arquitecturas, por lo que la **ronda final reduce el espacio de búsqueda** a
los dos candidatos Focal y los lleva a **100 épocas** para una comparación justa y
definitiva.

**Tabla comparativa (validación, 100 épocas):**

| Experimento | Modelo | Balanceo | Épocas | Final Train Loss | Val mAP@0.5 |
|---|---|---|---|---|---|
| **Exp 2 — FasterRCNN Focal** 🏆 | Faster R-CNN | Focal Loss | 100 | 0.0082 | **0.7489** |
| Exp 4 — RetinaNet Focal | RetinaNet | Focal Loss | 100 | 0.0163 | 0.7151 |

> **Ajuste del scheduler (decisión justificada):** al pasar a 100 épocas, se cambió
> el `StepLR` de `step_size=10` a `step_size=25` (con `gamma=0.5`). Con paso 10, a 100
> épocas el learning rate decae 10 veces y se vuelve insignificante a mitad del
> entrenamiento, estancando el modelo. Con paso 25 decae solo 4 veces, manteniendo un
> lr útil hasta el final. La curva de validación (`dev/curvas_aprendizaje_val_mAP.png`)
> lo confirma: ambos modelos siguen mejorando de forma estable hasta la época 100, sin
> aplanarse.

### 3.4. Comparación del modelo ganador contra YOLOv8

> **Nota:** YOLOv8 es la arquitectura de detección vista en el cursado. El ganador
> definido es **Faster R-CNN + Focal Loss**, así que la comparación relevante es la de
> §3.4.a. Se conserva también §3.4.b (RetinaNet) como referencia del candidato que
> compitió en la ronda final.

**Por qué se exploró torchvision (Faster R-CNN / RetinaNet) en lugar de quedarse
con YOLOv8:**

- **Control fino de la función de pérdida y de la cabeza:** torchvision permite
  intervenir directamente la `classification_head` / `fastrcnn_loss` para inyectar
  Focal Loss con γ y α propios, que es justamente la palanca para nuestro
  desbalance. En YOLO ese control es más indirecto.
- **Integración nativa con el pipeline PyTorch** ya construido (Dataset propio,
  `collate_fn`, métricas con torchmetrics) sin depender del framework
  `ultralytics` y su propio formato de entrenamiento.
- **Requisito de la cátedra:** se exige *fine-tuning* propio de un modelo
  preentrenado, no el uso directo de un modelo. Reemplazar y reentrenar la cabeza
  sobre torchvision deja explícito y trazable ese fine-tuning.

#### 3.4.a. Faster R-CNN + Focal Loss vs YOLOv8

| Criterio | Faster R-CNN + Focal (nuestro) | YOLOv8 |
|---|---|---|
| Arquitectura | **Dos etapas** (RPN + ROI heads) | Una etapa |
| Precisión de localización | **Mayor** en objetos pequeños/solapados | Buena, algo menor en cajas chicas |
| Patrones finos (mechas delgadas) | **Favorecido** por el refinamiento de la 2.ª etapa | Puede perder detalle |
| Tratamiento del desbalance | Focal Loss inyectada en la cabeza | Loss propia, menos ajustable |
| Velocidad / tamaño | Más lento y pesado (~158 MB) | Más rápido y liviano |
| Control para fine-tuning | **Alto** (cabeza y loss reemplazables) | Medio (vía `ultralytics`) |

**Por qué se eligió (si resulta ganador):** la **doble etapa** prioriza la
**precisión de localización**, decisiva cuando el patrón depende de detalles
geométricos finos (la relación cuerpo/mecha) y de cajas relativamente pequeñas
dentro del gráfico. La Focal Loss inyectada compensa su principal punto débil
frente a YOLO (la robustez ante el desbalance), quedándose con lo mejor de ambos:
localización precisa + foco en clases difíciles. El costo es un modelo más pesado y
lento, asumible para una app de inferencia bajo demanda.

#### 3.4.b. RetinaNet + Focal Loss vs YOLOv8

| Criterio | RetinaNet + Focal (nuestro) | YOLOv8 |
|---|---|---|
| Arquitectura | **Una etapa** (densa, anchors) | Una etapa |
| Tratamiento del desbalance | **Focal Loss nativa** (γ=2.0, α=0.25) | Loss propia, menos ajustable |
| Precisión de localización | Buena, comparable a YOLO | Buena |
| Velocidad / tamaño | Liviano y rápido (1 etapa) | Liviano y rápido |
| Control para fine-tuning | **Alto** (cabeza y γ/α reemplazables) | Medio (vía `ultralytics`) |
| Integración con el pipeline | Nativa torchvision/PyTorch | Framework aparte |

**Por qué se eligió (si resulta ganador):** RetinaNet ofrece el **mismo paradigma
de una etapa que YOLOv8** —liviano y rápido— pero con la **Focal Loss como
tratamiento de primera clase del desbalance**, que es exactamente nuestro problema,
y con el control total que da torchvision sobre los coeficientes γ/α de esa
pérdida. Es decir: las ventajas de eficiencia de YOLO, más una palanca explícita y
ajustable para las clases minoritarias, integrada de forma nativa en el pipeline
PyTorch del proyecto.

### 3.5. Métricas finales en test y análisis de errores

El modelo ganador (**Faster R-CNN + Focal Loss**, 100 épocas) se evaluó sobre el
**conjunto de test** (115 imágenes, nunca visto en entrenamiento ni validación) con
`torchmetrics`:

| Métrica global (test) | Valor |
|---|---|
| **mAP@0.5** | **0.8650** |
| **mAP@0.5:0.95** | **0.7305** |
| AR@100 (recall promedio) | 0.8807 |

**AP@0.5:0.95 por clase:**

| Clase | AP | | Clase | AP |
|---|---|---|---|---|
| Bearish Insidebar | 0.879 | | Bullish Insidebar | 0.741 |
| Hammer | 0.851 | | Bearish Engulfing | 0.627 |
| Bullish Engulfing | 0.731 | | Inverted Hammer | 0.554 |

**Comparación contra la ronda anterior (el efecto de la Focal Loss):** frente al mejor
modelo de v1 (Faster R-CNN + WeightedRandomSampler, 30 épocas: test mAP@0.5 **0.7714**,
mAP@0.5:0.95 0.4939), la Focal Loss a 100 épocas mejora **+9,4 puntos** de mAP@0.5 y
**+23,7** de mAP@0.5:0.95. Lo más relevante: las mayores mejoras se dan en las **clases
minoritarias** —*Bearish Engulfing* (0.382 → 0.627, **+24,5**) e *Inverted Hammer*
(0.360 → 0.554, **+19,4**)— que es **exactamente el efecto teórico esperado** de la
Focal Loss (reducir el peso de los ejemplos fáciles/mayoritarios para enfocar el
aprendizaje en los difíciles). Esto valida con datos la elección del balanceo.

**Análisis de errores:** las clases con menor AP siguen siendo las dos minoritarias
(*Inverted Hammer*, *Bearish Engulfing*), coherente con que son las de menos muestras;
y los dos *insidebar*, visualmente similares, son los pares más confundibles. El script
de entrenamiento (`dev/NotebookPython.py`) incluye una rutina que genera una grilla
*ground truth* vs predicción para inspección visual de falsos positivos/negativos.

### 3.6. Guardado del modelo

Los pesos del modelo final se exportan como **`dev/modelo.pth`** (state_dict). Como
el archivo supera los **100 MB** (≈158 MB) y excede el límite de GitHub, se versiona
con **Git LFS** (`git lfs track 'dev/*.pth'`, `.gitattributes`). Los checkpoints
de los experimentos intermedios (`{exp_name}_best.pth`) quedan documentados en el
notebook.

> **Estado del artefacto:** `dev/modelo.pth` es el **modelo ganador** (Faster R-CNN +
> Focal Loss, 100 épocas, test mAP@0.5 0.8650) y es el que sirve la app. En el
> despliegue (Streamlit Cloud, que no resuelve Git LFS) el mismo modelo se descarga
> desde `MODEL_URL` al iniciar.

---

## 4. Aplicación y despliegue

### 4.1. Arquitectura de la app y separación de responsabilidades

La app (carpeta [`prod/`](prod/)) está construida en **Streamlit** y separa
estrictamente responsabilidades, tal como pide la consigna:

- **[`prod/app.py`](prod/app.py) — capa de presentación:** interfaz Streamlit con un
  **tema claro, formal y sobrio** (hero, fila de *stats*, tarjetas, leyenda de colores
  por clase, dibujo de los *bounding boxes*, panel educativo por patrón y *disclaimer*
  de "no es recomendación de inversión"). Explica **todos** los patrones detectados (no
  solo el de mayor score) e incluye una sección de **limitaciones** de uso. Este archivo
  **no** contiene lógica de modelo.
- **[`prod/utils.py`](prod/utils.py) — lógica auxiliar:** carga del modelo,
  preprocesamiento de la imagen de entrada, inferencia y postprocesamiento de las
  detecciones (filtrado por umbral, mapeo de índices a nombres, agregación de score
  por clase).

### 4.2. Decisión clave: preprocesamiento de inferencia idéntico a val/test

`utils.preprocess_image` **replica exactamente** el `__getitem__` de validación/test
del notebook:

```
Image → convert_to_rgb (compone RGBA sobre fondo negro) → resize(224, 224) → to_tensor
```

**Por qué NO se aplica `Normalize` manual:** Faster R-CNN normaliza **internamente**
con su `GeneralizedRCNNTransform` (media/desvío de COCO). Aplicar un `Normalize`
extra en la app **duplicaría** la normalización y rompería silenciosamente el modelo
(uno de los errores más comunes y difíciles de detectar). Mantener el preprocesamiento
de producción **bit a bit igual** al de entrenamiento es la decisión que garantiza
que el modelo se comporte en la app como en el test.

### 4.3. Acondicionamiento opcional de la imagen de entrada

El modelo se entrenó con velas sobre **fondo negro** (§2.1). Las capturas reales del
usuario suelen tener **fondo blanco/grilla**, fuera de ese dominio. Por eso la app
ofrece un *checkbox* (activado por defecto) que aplica `remove_white_background` —el
**mismo algoritmo** del notebook de preparación— para llevar la imagen subida al
dominio de entrenamiento **antes** de inferir. Es un acondicionamiento de la entrada;
**no** altera el preprocesamiento de val/test, que se aplica igual después.

La barra lateral también expone un **slider de umbral de confianza** (default 0.5):
subirlo muestra solo detecciones muy seguras; bajarlo, más detecciones.

**Detección de múltiples patrones (NMS por clase):** el modelo trae un *Non-Max
Suppression* interno (`box_nms_thresh=0.3`) que suprime cajas solapadas **sin importar
la clase**, dejando un solo patrón por zona. Como distintos patrones pueden ocupar la
misma región del gráfico, en inferencia se **desactiva el NMS interno** (en runtime, sin
tocar los pesos) y se aplica un **NMS por clase** en `predict()`: se eliminan duplicados
solo dentro de la **misma** clase. Así, patrones de clases distintas que se solapan
**conviven** y se muestran y explican todos.

### 4.4. Carga del modelo y caché

- La carga está **cacheada con `@st.cache_resource`**, de modo que el modelo
  (~158 MB) se cargue **una sola vez** por sesión del servidor y no se recargue en
  cada interacción del usuario.
- **Estrategia de carga robusta** (`_ensure_model_file`), pensada para local y
  cloud:
  1. Si existe `dev/modelo.pth` localmente (repo clonado con Git LFS), se usa.
  2. Si no, y hay una variable **`MODEL_URL`** configurada (Hugging Face, GitHub
     Release o Google Drive), se **descarga** a una caché local con barra de
     progreso.
  3. Si no hay ninguna de las dos, se lanza un error claro y accionable.

  Esto resuelve que **Streamlit Community Cloud no resuelve bien los punteros de
  Git LFS**: en ese entorno el modelo se baja desde `MODEL_URL`.

### 4.5. Despliegue

- **Hosting:** Streamlit Community Cloud (gratuito, integrado con GitHub), accesible
  públicamente sin login ni instalación:
  <https://reconocimiento-patrones-velas-japonesas.streamlit.app/>
- **`prod/requirements.txt` con versiones fijadas** para un despliegue reproducible
  (`streamlit==1.58.0`, `torch==2.11.0`, `torchvision==0.26.0`, `pillow`, `pandas`,
  `numpy`, `opencv-python-headless`). En un hosting sin GPU como Streamlit Cloud se usa
  la *build* **CPU** de `torch`/`torchvision` (la app detecta CPU/GPU automáticamente en
  runtime), y la descarga del modelo desde Google Drive usa `gdown`.

### 4.6. Comportamiento esperado en la demo

- **Casos esperados** (gráficos de velas representativos): el modelo dibuja la(s)
  caja(s) del patrón con su score y muestra el panel de sesgo alcista/bajista.
- **Casos límite** (imágenes que no son gráficos, baja calidad, fondo no estándar):
  si ninguna detección supera el umbral, la app informa que **no se reconoció ningún
  patrón con claridad** y sugiere otra captura o ajustar el umbral — comportamiento
  honesto frente a entradas fuera de dominio.

---

## 5. Cierre: resumen de decisiones y justificaciones

| Decisión | Qué se eligió | Por qué |
|---|---|---|
| **Tipo de problema** | Detección de objetos (no clasificación) | Hay que **localizar** cada patrón dentro del gráfico, no solo etiquetar la imagen |
| **Red neuronal vs reglas** | Fine-tuning de modelo de detección | Las reglas codificadas no generalizan a cambios de escala, color y fondo; la red aprende la morfología abstracta |
| **Dominio de imágenes** | Remoción de fondo → fondo negro | Unifica el dominio y se replica idéntico en inferencia (coherencia train↔prod) |
| **Particionado** | Estratificado 70/20/10, `seed=42` | Conserva la proporción de clases (dataset chico y desbalanceado) y es reproducible |
| **Augmentation** | Solo ColorJitter en train | Robustez a iluminación/color sin romper la semántica; flips/rotaciones invertirían el sesgo del patrón |
| **Transfer learning** | Pesos COCO + reemplazo de cabeza | Acelera la convergencia con pocos datos; adapta la salida a 6 clases |
| **Fine-tuning** | Full fine-tuning (sin congelar) | Dominio muy distinto a COCO; el dataset homogéneo permite readaptar toda la red |
| **Balanceo** | Focal Loss (γ=2.0, α=0.25) | Enfoca el aprendizaje en clases difíciles/minoritarias; supera al baseline y al sampler |
| **Modelo final** | Faster R-CNN + Focal Loss (ganó a RetinaNet Focal) | Mejor mAP en val y test (0.8650@0.5); la 2-etapas da localización precisa y la Focal trata el desbalance |
| **Framework de detección** | torchvision (no ultralytics/YOLOv8) | Control fino de loss y cabeza, integración nativa con el pipeline PyTorch, fine-tuning trazable |
| **Preprocesamiento en prod** | Idéntico a val/test, sin Normalize manual | Faster R-CNN normaliza internamente; duplicarlo rompería el modelo |
| **Carga del modelo** | `@st.cache_resource` + LFS/`MODEL_URL` | Evita recargar 158 MB por interacción; soluciona LFS en Streamlit Cloud |
| **Hosting** | Streamlit Community Cloud | Gratuito, integrado con GitHub, público sin login |

> **Resultado final:** el modelo ganador es **Faster R-CNN + Focal Loss** (100 épocas),
> con **test mAP@0.5 = 0.8650** (vs 0.7714 del mejor de la ronda previa). Es el modelo
> guardado en `dev/modelo.pth` y el que sirve la app desplegada. La Focal Loss mejoró
> especialmente las clases minoritarias, validando con datos la estrategia de balanceo.
