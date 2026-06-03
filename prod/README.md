# prod/

Carpeta reservada para el código de inferencia y servicio en producción del modelo entrenado.

Por el momento está vacía: en esta etapa del proyecto solo se completó la preparación del dataset.

Contenido previsto en futuras etapas:
- Script de inferencia (`predict.py`) que carga el modelo entrenado y clasifica una imagen nueva.
- Modelo serializado (pesos `.pt` o `.pth`).
- Eventual API REST / interfaz para servir el modelo.
