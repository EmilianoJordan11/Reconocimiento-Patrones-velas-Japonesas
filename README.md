# Reconocimiento de Patrones de Velas Japonesas

Proyecto de clasificación de imágenes con redes neuronales para reconocer **6 patrones de velas japonesas** usados en análisis técnico de mercados financieros.

## Patrones reconocidos

1. Bearish Engulfing
2. Bearish Insidebar
3. Bullish Engulfing
4. Bullish Insidebar
5. Hammer
6. Inverted Hammer

## Estructura del repositorio

```
.
├── data/                              # CSVs de splits + README del dataset
│   ├── README.md
│   ├── train.csv                      # 813 muestras (70%)
│   ├── val.csv                        # 232 muestras (20%)
│   ├── test.csv                       # 115 muestras (10%)
│   ├── raw/                           # Dataset original (no versionado)
│   └── processed/                     # Dataset con fondo removido (no versionado)
├── dev/                               # Notebooks de desarrollo
│   └── 01_dataset_preparation.ipynb
├── prod/                              # (Placeholder) código de inferencia futuro
├── scripts/                           # Scripts auxiliares
│   └── create_splits.py
├── dowload_dataset.py                 # Descarga del dataset desde Roboflow
├── requirements.txt                   # Dependencias Python
├── planning.md                        # Plan de implementación
└── contexto.md                        # Contexto del proyecto
```

## Dataset

- **Fuente:** [Roboflow — Candlestick Pattern Dataset v1](https://universe.roboflow.com/madhumitha-jc-hvsdd/candlestick-pattern/dataset/1)
- **Licencia:** CC BY 4.0
- **Formato original:** YOLOv8 (adaptado para clasificación)
- **Total:** 1.160 imágenes distribuidas en 6 clases

## Cómo correr el proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/EmilianoJordan11/Reconocimiento-Patrones-velas-Japonesas.git
cd Reconocimiento-Patrones-velas-Japonesas
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar el notebook de preparación

```bash
jupyter notebook dev/01_dataset_preparation.ipynb
```

El notebook descarga el dataset, lo procesa, genera los splits y deja todo listo para entrenar.

## Estado actual

- [x] Descarga reproducible del dataset
- [x] Preprocesamiento (eliminación de fondo blanco)
- [x] Splits estratificados 70/20/10 con seed fijo
- [x] Clase `Dataset` personalizada de PyTorch
- [x] DataLoaders configurados
- [x] Data augmentation (ColorJitter) en train
- [x] Visualizaciones de control
- [ ] Selección y entrenamiento del modelo
- [ ] Evaluación y métricas
- [ ] Despliegue
