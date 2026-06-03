import os
import shutil
from roboflow import Roboflow

rf = Roboflow(api_key="2HWZKz4mE7I0TJUj4Aea")

project = rf.workspace("madhumitha-jc-hvsdd").project("candlestick-pattern")

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")

print("Descargando el dataset en la carpeta temporal de Roboflow...")
dataset = project.version(1).download("yolov8")

source_dir = dataset.location
if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)

destination_dir = os.path.join(data_dir, "raw")

if os.path.abspath(source_dir) != os.path.abspath(destination_dir):
    print(f"Moviendo dataset desde {source_dir} a {destination_dir}")
    if os.path.exists(destination_dir):
        print(f"El destino {destination_dir} ya existe. Eliminando antes de mover...")
        shutil.rmtree(destination_dir)
    shutil.move(source_dir, destination_dir)
    print("Descarga movida correctamente a la carpeta data/raw.")
else:
    print("El dataset ya está en la carpeta data/raw.")

print(f"Ubicación final del dataset: {destination_dir}")