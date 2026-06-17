# prod/ — App de inferencia (Semana 4)

App web en **Streamlit** que sirve el modelo entrenado en `dev/` para **detectar
patrones de velas japonesas** en una imagen.

> ⚠️ **Importante:** el modelo es un **detector de objetos** (Faster R-CNN
> ResNet50-FPN), no un clasificador de imagen única. Por cada imagen devuelve
> un conjunto de **detecciones**, cada una con su *bounding box*, su **clase**
> (uno de los 6 patrones) y un **score de confianza**.

## Contenido

| Archivo | Descripción |
|---|---|
| `app.py` | Interfaz Streamlit: subir imagen → ver cajas detectadas + tabla de detecciones + confianza por patrón. |
| `utils.py` | Carga del modelo (cacheada con `@st.cache_resource`), preprocesamiento idéntico a val/test e inferencia/postprocesamiento. |
| `requirements.txt` | Dependencias con versiones fijadas. |

## Clases detectadas (6)

`Bearish Engulfing`, `Bearish Insidebar`, `Bullish Engulfing`,
`Bullish Insidebar`, `Hammer`, `Inverted_Hammer`.

## Pipeline de inferencia (idéntico a val/test)

El preprocesamiento replica exactamente el `__getitem__` del dataset de
validación/test del notebook `dev/02_model_training.ipynb`:

1. Convertir a RGB. Si la imagen es **RGBA**, se compone sobre **fondo negro**
   usando el canal alpha como máscara.
2. `resize((224, 224))` con PIL (resize directo, sin preservar *aspect ratio*).
3. `torchvision.transforms.functional.to_tensor` → tensor en `[0, 1]`.
4. **Sin `Normalize` manual**: Faster R-CNN normaliza internamente con su propio
   `GeneralizedRCNNTransform` (medias/desvíos de COCO).

El postprocesamiento aplica un umbral de confianza configurable (por defecto
`0.5`), ordena las detecciones por score y agrega el score máximo por clase.

## Cómo correr la app localmente

Desde la **raíz del repositorio**:

```bash
# 1. Crear/activar un entorno virtual (opcional pero recomendado)
python -m venv venv
# Windows:  venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 2. Instalar dependencias
pip install -r prod/requirements.txt

# 3. Asegurar que el modelo esté disponible (ver sección siguiente)
git lfs install
git lfs pull            # trae dev/modelo.pth (~158 MB) desde Git LFS

# 4. Lanzar la app
streamlit run prod/app.py
```

La app abre en `http://localhost:8501`.

## El modelo pesa 158 MB (> límite de GitHub) — estrategia de carga

`dev/modelo.pth` pesa **~158 MB**, por encima del límite de **100 MB** de
GitHub. Además, varios hostings (entre ellos **Streamlit Community Cloud**) no
resuelven los punteros de **Git LFS**: al clonar bajan el archivo de texto que
apunta al binario, no el binario real, y la carga del modelo falla.

`utils.py` implementa una estrategia con **fallback** (función
`_ensure_model_file()`):

1. **Local:** si existe `dev/modelo.pth` y pesa > 1 MB (descarta el puntero LFS
   sin resolver), lo usa directamente.
2. **Remoto:** si no está disponible localmente y la variable de entorno
   `MODEL_URL` está definida, **descarga** el modelo a una caché local
   (`prod/modelo.pth`) mostrando una barra de progreso.
3. **Error accionable:** si no hay ninguna de las dos, lanza un mensaje
   explicando cómo resolverlo.

### Configurar la descarga remota (para deploy)

Subí `modelo.pth` a un almacenamiento con **descarga directa** y definí
`MODEL_URL`:

- **Hugging Face Hub** (recomendado):
  `https://huggingface.co/<usuario>/<repo>/resolve/main/modelo.pth`
- **GitHub Release** (asset de un release, no cuenta para el límite del repo):
  `https://github.com/<owner>/<repo>/releases/download/<tag>/modelo.pth`
- **Google Drive:** usá un enlace de descarga directa (con `confirm`) o la
  librería `gdown`.

En **Streamlit Community Cloud**, agregá `MODEL_URL` en
*App → Settings → Secrets*. Localmente:

```bash
# Windows (PowerShell)
$env:MODEL_URL = "https://.../modelo.pth"
# Linux/Mac
export MODEL_URL="https://.../modelo.pth"
```

### Alternativa: Git LFS

Si el hosting **sí** resuelve Git LFS (o se despliega vía Docker con
`git lfs pull` en el build), no hace falta `MODEL_URL`: el archivo local se usa
directamente. Este repo ya tiene `dev/*.pth` rastreado por LFS
(ver `.gitattributes`).

## Despliegue en CPU (sin GPU)

La app detecta automáticamente CPU/GPU. Para un hosting sin GPU, instalá la
build **CPU** de PyTorch (ver nota en `prod/requirements.txt`):

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.11.0
torchvision==0.26.0
```
