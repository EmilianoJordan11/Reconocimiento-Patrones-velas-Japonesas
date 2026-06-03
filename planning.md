## Plan: Preparación del dataset PyTorch sin data augmentation

TL;DR: completar el notebook y la documentación para cargar el dataset YOLO en PyTorch, generar splits reproducibles, aplicar preprocesamiento y verificar los DataLoader sin implementar data augmentation.

**Pasos**
1. Documentar la fuente y la estructura del dataset.
   - Crear `data/README.md` con origen del dataset, URL de Roboflow/Kaggle y la estructura esperada para `data/raw/` y `data/processed/`.
   - Explicar que `data/processed/dataset_no_background/` contiene imágenes PNG procesadas y etiquetas YOLO en archivos `.txt`.
   - Añadir en el notebook una celda introductoria que describa esta estructura.

2. Integrar la descarga y el preprocesamiento reproducible.
   - Verificar que `dowload_dataset.py` descarga el dataset y lo deja en `data/raw/dataset/`.
   - En el notebook, incluir una celda con instrucciones claras o un llamado al script de descarga.
   - Asegurar que el notebook puede correrse de punta a punta tras clonar el repositorio (instalar dependencias → descargar dataset → ejecutar notebook).

3. Corregir y/o regenerar los CSV de splits.
   - Detectar el particionado actual deficiente, con valid/test solo de la clase 4.
   - Generar splits balanceados y reproducibles con una semilla fija (`seed = 42` o similar).
   - Guardar los resultados en `data/train.csv`, `data/val.csv`, `data/test.csv`.
   - Documentar la cantidad de imágenes y la distribución por clase en cada split.
   - Usar un criterio de particionado estratificado sobre clases para mantener balance.

4. Implementar la carga de datos PyTorch.
   - Añadir en el notebook una clase `CandlestickDataset` que herede de `torch.utils.data.Dataset`.
   - La clase debe leer filas de los CSV, cargar cada imagen y su etiqueta correspondiente.
   - Incluir un método para transformar imágenes RGBA a RGB si es necesario.
   - Justificar por qué `ImageFolder` no es adecuado: el dataset tiene etiquetas YOLO en `.txt` por imagen y no sigue una jerarquía `class/image.png`.

5. Configurar DataLoaders.
   - Crear `DataLoader` para train, valid y test con `batch_size` apropiado (por ejemplo 16 o 32).
   - Usar `shuffle=True` solo en train, `shuffle=False` en valid y test.
   - Definir `num_workers` razonable según el entorno.

6. Definir preprocesamiento sin augmentations.
   - Train/valid/test deben recibir las mismas transformaciones base salvo shuffle.
   - Aplicar resize a `224x224` u otro tamaño adecuado según el modelo preentrenado elegido.
   - Convertir a tensor y normalizar con los valores de media y desviación estándar de ImageNet.
   - Documentar la razón de las transformaciones y el manejo de RGBA.

7. Verificación final en el notebook.
   - Cargar un batch del DataLoader de train.
   - Mostrar imágenes desnormalizadas con sus etiquetas.
   - Reportar dimensiones de tensores `(batch_size, canales, alto, ancho)` y rango de valores tras normalización.
   - Confirmar visualmente que imagen y etiqueta coinciden.

8. Asegurar la estructura del repositorio.
   - Confirmar `data/`, `dev/`, `prod/` presentes.
   - Mantener notebook en `dev/01_dataset_preparation.ipynb`.
   - No versionar imágenes: `.gitignore` ya cubre `data/raw/`, `data/processed/` y extensiones de imagen.
   - Versionar solo CSV de splits y documentación liviana.

**Archivos relevantes**
- `c:\Users\matia\OneDrive\Documentos\RedesNeuronales\dev\01_dataset_preparation.ipynb`
- `c:\Users\matia\OneDrive\Documentos\RedesNeuronales\dowload_dataset.py`
- `c:\Users\matia\OneDrive\Documentos\RedesNeuronales\requirements.txt`
- `c:\Users\matia\OneDrive\Documentos\RedesNeuronales\data\README.md` (nuevo)
- `c:\Users\matia\OneDrive\Documentos\RedesNeuronales\.gitignore`

**Verificación**
1. Correr el notebook en un entorno limpio tras clonar el repo.
2. Confirmar que los CSV contienen rutas relativas correctas y que `CandlestickDataset` carga imágenes/etiquetas.
3. Revisar que los DataLoaders devuelven tensores con el formato esperado.
4. Validar que el notebook documenta claramente la estructura de datos, el particionado y el preprocesamiento.

**Decisiones**
- No implementar data augmentation en esta etapa.
- Usar un dataset personalizado de PyTorch porque la etiqueta es YOLO por imagen y no se puede usar `ImageFolder`.
- Corregir el particionado existente hacia un split balanceado y reproducible.