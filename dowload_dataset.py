"""
Script de descarga del dataset de patrones de velas japonesas desde Roboflow.

Este script:
1. Se conecta a Roboflow usando la API key del proyecto.
2. Descarga el dataset en formato YOLOv8 (versión 1).
3. Mueve la descarga a `data/raw/` para mantener una estructura predecible.

Uso:
    python dowload_dataset.py

Fuente del dataset:
    https://universe.roboflow.com/madhumitha-jc-hvsdd/candlestick-pattern/dataset/1
"""

import os
import shutil
from roboflow import Roboflow

# Inicializamos el cliente de Roboflow con la API key del workspace.
# NOTA: idealmente esta key debería venir de una variable de entorno y no estar hardcodeada.
rf = Roboflow(api_key="2HWZKz4mE7I0TJUj4Aea")

# Navegamos hasta el proyecto específico dentro del workspace de Roboflow.
project = rf.workspace("madhumitha-jc-hvsdd").project("candlestick-pattern")

# Calculamos rutas absolutas relativas al script para que funcione desde cualquier
# directorio donde se ejecute (ej: `python dowload_dataset.py` o `python /ruta/al/script.py`).
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")

# Roboflow descarga el dataset en una carpeta temporal con nombre tipo "candlestick-pattern-1".
# Esa carpeta hay que moverla a `data/raw/` para mantener una estructura predecible
# que el resto del pipeline (notebook, splits) pueda usar.
print("Descargando el dataset en la carpeta temporal de Roboflow...")
dataset = project.version(1).download("yolov8")

source_dir = dataset.location  # Carpeta donde Roboflow dejó la descarga.

# Aseguramos que `data/` exista antes de mover nada adentro.
if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)

destination_dir = os.path.join(data_dir, "raw")

# Solo movemos si la descarga quedó en una ubicación distinta a la deseada.
# (Si Roboflow ya descargó directamente en data/raw/, no hacemos nada).
if os.path.abspath(source_dir) != os.path.abspath(destination_dir):
    print(f"Moviendo dataset desde {source_dir} a {destination_dir}")

    # Si ya existe una descarga previa, la borramos para evitar mezclar archivos.
    if os.path.exists(destination_dir):
        print(f"El destino {destination_dir} ya existe. Eliminando antes de mover...")
        shutil.rmtree(destination_dir)

    shutil.move(source_dir, destination_dir)
    print("Descarga movida correctamente a la carpeta data/raw.")
else:
    print("El dataset ya está en la carpeta data/raw.")

print(f"Ubicación final del dataset: {destination_dir}")
