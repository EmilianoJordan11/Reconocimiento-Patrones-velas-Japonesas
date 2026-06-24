# Presentación — Semana 4: Interfaz y Despliegue

**Proyecto:** Reconocimiento de Patrones de Velas Japonesas
**App:** *Predictor* — detector de patrones en gráficos de velas
**Modelo:** Faster R-CNN · ResNet50-FPN (detección de objetos, no clasificación simple)

> **Recordatorio:** la nota es individual. Cada integrante debe poder explicar y defender
> cualquier parte: el código de la app, la carga del modelo y el flujo de inferencia.

---

## 0. Resumen en una frase

El usuario sube la imagen de un gráfico de velas; la app la lleva al mismo dominio
con el que se entrenó el modelo, ejecuta un **detector Faster R-CNN** que localiza
los patrones con su *bounding box* y score, y devuelve la imagen anotada + una
explicación en lenguaje natural de qué patrón apareció y qué suele indicar.

---

## a) Aplicación web

### Demostración en vivo (guion sugerido)
1. **Abrir la URL pública** (sin login ni instalación).
2. **Caso esperado:** subir una imagen representativa del dataset (un *Hammer*, un
   *Bullish Engulfing*, etc.). Mostrar:
   - la imagen analizada y la imagen con el **bounding box** dibujado,
   - la **clase predicha** y la **confianza** (score),
   - la explicación educativa del patrón y el resumen de **confianza por patrón** (las 6 clases).
3. **Caso límite — imagen fuera de dominio:** subir algo que no es un gráfico de velas
   (una foto cualquiera). La app responde *"No se reconoció ningún patrón con claridad"*
   en vez de inventar una predicción.
4. **Caso límite — captura real con fondo blanco/grilla:** mostrar el checkbox
   **"Quitar fondo claro"**. El modelo se entrenó con velas sobre **fondo negro**;
   una captura con fondo blanco está fuera de dominio. Activar la opción lleva la
   imagen al dominio de entrenamiento y mejora la detección.
5. **Control de sensibilidad:** mover el **umbral de confianza** (slider) para mostrar
   cómo aparecen/desaparecen detecciones poco seguras.

### Cómo se comporta el modelo en cada caso (para comentar oralmente)
- **Imágenes representativas:** detecta el patrón con score alto y lo localiza bien.
- **Fuera de clase:** no fuerza una predicción; si nada supera el umbral, lo dice.
- **Baja calidad / fondo no esperado:** puede fallar; por eso existe el pre-acondicionado
  de fondo y el umbral ajustable. Es honesto reconocer que el dominio de entrenamiento
  (velas sintéticas sobre fondo negro) limita la generalización a capturas reales.

### Interfaz
Funcional y clara (no se evalúa diseño avanzado). Muestra la información útil:
**imagen ingresada, imagen anotada con la caja, clase predicha, confianza y confianza
por clase**. Tecnología: **Streamlit** (la recomendada para el proyecto).

---

## b) Código de la aplicación

### Separación de responsabilidades
- **`prod/app.py`** → capa de **presentación**: layout, widgets (uploader, slider,
  checkbox), dibujo de cajas, textos educativos. No contiene lógica del modelo.
- **`prod/utils.py`** → funciones auxiliares:
  - **carga del modelo** (`load_model`),
  - **preprocesamiento** de la entrada (`preprocess_image`),
  - **inferencia + postprocesamiento** (`predict`).

### Preprocesamiento IDÉNTICO a validación/test  *(punto crítico)*
El pipeline de producción replica exactamente el `__getitem__` de val/test del
notebook de entrenamiento:

```
Image  ->  convert_to_rgb  ->  resize((224, 224))  ->  F.to_tensor
```

- Mismo **resize a 224×224** (`IMAGE_SIZE`).
- **No** se aplica `Normalize` manual: Faster R-CNN normaliza internamente con su
  propio `GeneralizedRCNNTransform` (media/desvío de COCO). Agregar un Normalize
  acá **rompería** la coherencia con el entrenamiento. (Defendible: este es uno de
  los errores silenciosos más comunes y lo evitamos a propósito.)
- `convert_to_rgb` compone RGBA sobre **fondo negro**, igual que en el dataset.

### Carga del modelo cacheada
`utils.load_model()` está decorada con **`@st.cache_resource`**: el modelo (~158 MB)
se carga **una sola vez** por sesión del servidor, no en cada interacción del usuario.
La app detecta **CPU/GPU** en runtime (`cuda` si está disponible, si no `cpu`).

### Flujo de inferencia (para defender de punta a punta)
1. Usuario sube imagen → `Image.open`.
2. (Opcional) `strip_white_background` lleva la imagen al dominio de entrenamiento.
3. `utils.predict`: preprocesa → `model([tensor])` → filtra por umbral → mapea
   `label_idx (1..6)` a nombre de clase (el 0 es fondo) → ordena por score.
4. `app.py` dibuja las cajas y muestra resultados + explicación.

---

## c) Despliegue

- **Plataforma:** Streamlit Community Cloud (gratuito, integrado con GitHub).
- **Acceso:** público, **sin login ni instalación**, vía URL. La URL queda en el
  `README.md` de la raíz.
- **`requirements.txt` con versiones fijadas** (reproducible). En `prod/requirements.txt`:
  - `streamlit==1.58.0`, `torch==2.11.0`, `torchvision==0.26.0`, `pillow==12.2.0`,
    `pandas==3.0.3`, `numpy==2.4.4`, `opencv-python-headless==4.13.0`, `gdown==5.2.0`.
  - Se usa la **build CPU** de torch/torchvision (`--extra-index-url` de PyTorch CPU),
    porque Streamlit Cloud **no tiene GPU**.

### Estrategia para el modelo que supera el límite de tamaño  *(documentar — punto d/c)*
`dev/modelo.pth` pesa **~158 MB**, por encima del límite de 100 MB de archivos en
GitHub, y los hostings tipo Streamlit Cloud **no resuelven los punteros de Git LFS**.
Estrategia adoptada:

- El modelo se aloja en **Google Drive** (compartido públicamente).
- En el arranque, si no hay un `modelo.pth` válido local, la app lo **descarga** desde
  la variable de entorno / secret **`MODEL_URL`** a una caché local.
- Para Drive se usa **`gdown`** (con `fuzzy=True`): los archivos >100 MB de Drive
  devuelven una página de confirmación de antivirus que un `urllib` simple bajaría
  como si fuese el binario, corrompiéndolo; `gdown` sortea ese paso.
- Validación: si la descarga devuelve un archivo < 1 MB (p. ej. un HTML de error),
  se aborta con un mensaje claro en vez de cachear basura.
- El `MODEL_URL` **no se versiona**: se define en el panel **Secrets** de Streamlit Cloud.

---

## d) Repositorio final

### Estructura (data / dev / prod)
```
.
├── data/            CSVs de splits (train/val/test) + README del dataset
├── dev/             Notebooks de desarrollo + pesos de los 3 experimentos + modelo.pth (Git LFS)
├── prod/            App de inferencia: app.py, utils.py, requirements.txt, README.md
├── .streamlit/      config.toml (tema) + secrets.toml.example
└── README.md        Descripción, integrantes, link a la app, instrucciones, dataset
```

### README.md de la raíz — debe incluir
- [x] Descripción del proyecto
- [ ] **Integrantes** *(completar nombres/legajos — hoy son placeholders)*
- [ ] **Link a la app desplegada** *(completar con la URL real de Streamlit Cloud)*
- [x] Instrucciones para clonar y correr localmente (con `git lfs pull`)
- [x] Link al dataset (Roboflow — Candlestick Pattern, CC BY 4.0)
- [x] Repo público

---

## Apéndice — Datos del modelo (por si preguntan)

- **Arquitectura:** Faster R-CNN, backbone ResNet50-FPN (preentrenado en COCO),
  cabeza adaptada a **6 patrones + fondo = 7 clases**.
- **Selección entre 3 experimentos** (`dev/02_model_training.ipynb`):

  | Exp | Modelo base | Balanceo | Val mAP@0.5 |
  |---|---|---|---|
  | 1 | Faster R-CNN | Baseline | ~0.74 |
  | 2 | Faster R-CNN | WeightedRandomSampler | **~0.76** ✅ |
  | 3 | RetinaNet | Focal Loss | ~0.69 |

- **Ganador:** Experimento 2 → `dev/modelo.pth`. **Test mAP@0.5 ≈ 0.77**.
- **6 clases:** Bearish Engulfing, Bearish Insidebar, Bullish Engulfing,
  Bullish Insidebar, Hammer, Inverted_Hammer.
- **Dataset:** Roboflow Candlestick Pattern v1, 1.160 imágenes, splits 70/20/10 (seed 42).

---

## ⚠️ Pendientes antes de exponer (checklist honesto)

- [ ] **Completar integrantes** en `README.md` (hoy dicen "Integrante 1/2/3").
- [ ] **Pegar la URL pública** de la app en `README.md` (hoy es un placeholder `TODO-link`).
- [ ] Confirmar que **Emiliano desplegó** la app y que la URL abre sin login.
- [ ] Tener a mano **2–3 imágenes de prueba**: 1 representativa + 1 fuera de clase + 1 captura real.
- [ ] Que **cada integrante** pueda explicar: preprocesamiento = val/test, `@st.cache_resource`,
      y la estrategia de descarga del modelo (Drive + gdown + secret).
