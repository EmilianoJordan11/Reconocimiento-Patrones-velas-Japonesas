# Dataset

## Origen

Este proyecto usa un dataset de patrones de velas (candlestick patterns) descargado desde Roboflow.

- Fuente: https://universe.roboflow.com/madhumitha-jc-hvsdd/candlestick-pattern/dataset/1
- Licencia: CC BY 4.0

## Estructura esperada

El repositorio no versiona las imágenes ni los archivos binarios grandes. El contenido que debe existir localmente es:

```
data/
  raw/
    dataset/
      data.yaml
      train/
        images/
        labels/
      valid/
        images/
        labels/
      test/
        images/
        labels/
  processed/
    dataset_no_background/
      data.yaml
      train/
        images/
        labels/
      valid/
        images/
        labels/
      test/
        images/
        labels/
  train.csv
  val.csv
  test.csv
```

## Descarga y uso

1. Instalar dependencias:

```bash
pip install -r requirements.txt
```

2. Descargar el dataset y procesarlo:

```bash
python dowload_dataset.py
```

3. Generar splits reproducibles (si no existen o se quiere regenerar):

```bash
python scripts/create_splits.py
```

4. Ejecutar el notebook de preparación de datos:

```bash
jupyter notebook dev/01_dataset_preparation.ipynb
```

## CSV de splits

Los archivos `data/train.csv`, `data/val.csv` y `data/test.csv` contienen rutas relativas a `data/processed/dataset_no_background`. Estos archivos se versionan porque son livianos y garantizan que todos trabajen con el mismo particionado.

## Nota

No se deben subir al repositorio las imágenes ni las carpetas `data/raw/` y `data/processed/`.
