# %% [markdown]
# Reescritura Estructurada Definitiva para Ejecución Segura en Terminal de Windows
# Soporta multiprocesamiento nativo (NUM_WORKERS = 4) y aislamiento de hilos de GPU.

# %%
import os
import random
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # Reservado estándar para funciones de NN
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.transforms import functional as TF  # Cambiado a TF para evitar colisiones
import yaml

# =====================================================================
# CONFIGURACIÓN DE RUTAS DINÁMICAS (Resuelve errores de directorio de trabajo)
# =====================================================================
CURRENT_DIR = Path(__file__).resolve().parent  # Carpeta dev/
ROOT = CURRENT_DIR.parent                      # Raíz del proyecto
DATA_ROOT = ROOT / "data"
PROCESSED_ROOT = DATA_ROOT / "processed" / "dataset_no_background"
VENV_BIN = CURRENT_DIR.parent / ".venv" / "Scripts"
VENV_LIB_SITE = CURRENT_DIR.parent / ".venv" / "Lib" / "site-packages" / "torch" / "lib"

if os.name == 'nt':  # Si estás en Windows
    if VENV_BIN.exists():
        os.environ["PATH"] = str(VENV_BIN) + os.pathsep + os.environ["PATH"]
    if VENV_LIB_SITE.exists():
        os.add_dll_directory(str(VENV_LIB_SITE))
SPLIT_CSV = {
    "train": DATA_ROOT / "train.csv",
    "valid": DATA_ROOT / "val.csv",
    "test": DATA_ROOT / "test.csv",
}

# --- Hiperparámetros Globales ---
IMAGE_SIZE = 224
BATCH_SIZE = 8       
NUM_EPOCHS = 100      
NUM_WORKERS = 4      
SEED = 42

# =====================================================================
# DEFINICIONES DE UTILIDADES Y ARQUITECTURAS (Nivel Global Seguro)
# =====================================================================

def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def convert_to_rgb(img: Image.Image) -> Image.Image:
    """Asegura la conversión de imágenes RGBA descomponiendo contra un fondo negro."""
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (0, 0, 0))
        background.paste(img, mask=img.split()[3])
        return background
    return img.convert("RGB") if img.mode != "RGB" else img

def read_yolo_annotations(label_path: Path) -> list:
    """Lee anotaciones en formato YOLO de archivos de texto."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                boxes.append((
                    int(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                ))
            except ValueError:
                continue
    return boxes

def yolo_to_xyxy(xc, yc, w, h, img_w=IMAGE_SIZE, img_h=IMAGE_SIZE):
    """Transforma coordenadas YOLO [0,1] a Pascal VOC absolutizadas [x1, y1, x2, y2]."""
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return [x1, y1, x2, y2]

def detection_collate_fn(batch):
    """Collate especial para evitar colapsos por cantidad variable de cajas por imagen."""
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)

class CandlestickDetectionDataset(Dataset):
    """Dataset personalizado para detección de patrones de velas japonesas."""
    def __init__(self, csv_path: Path, dataset_root: Path, label_offset=0, transform=None):
        self.df = pd.read_csv(csv_path)
        self.dataset_root = dataset_root
        self.label_offset = label_offset
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.dataset_root / row["image"]
        label_path = self.dataset_root / row["label"]

        img_pil = Image.open(image_path)
        img_pil = convert_to_rgb(img_pil)
        img_pil = img_pil.resize((IMAGE_SIZE, IMAGE_SIZE))

        annotations = read_yolo_annotations(label_path)
        boxes_list = []
        labels_list = []

        for cls_id, xc, yc, bw, bh in annotations:
            bbox = yolo_to_xyxy(xc, yc, bw, bh, IMAGE_SIZE, IMAGE_SIZE)
            boxes_list.append(bbox)
            labels_list.append(cls_id + self.label_offset)

        if boxes_list:
            boxes = torch.tensor(boxes_list, dtype=torch.float32)
            labels = torch.tensor(labels_list, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)

        img_tensor = TF.to_tensor(img_pil)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }
        return img_tensor, target

def build_faster_rcnn_model(num_classes=7):
    """Construye un modelo Faster R-CNN con backbone ResNet50 preentrenado."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="COCO_V1",
        box_nms_thresh=0.3
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    for param in model.parameters():
        param.requires_grad = True
    return model

def softmax_focal_loss(inputs, targets, gamma=2.0, alpha=0.25):
    """Focal loss multiclase usando softmax. Reduce a cross-entropy cuando gamma=0."""
    ce = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce)
    return (alpha * (1 - pt) ** gamma * ce).mean()

def build_retinanet_model(num_classes=6, focal_gamma=2.0, focal_alpha=0.25):
    """Construye un modelo RetinaNet parametrizando los coeficientes de Focal Loss."""
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights="COCO_V1")
    in_channels = model.head.classification_head.conv[0][0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels, num_anchors, num_classes=num_classes
    )
    model.head.classification_head.focal_loss_gamma = focal_gamma
    model.head.classification_head.focal_loss_alpha = focal_alpha
    
    for param in model.parameters():
        param.requires_grad = True
    return model

def train_detection_model(model, train_loader, valid_loader, optimizer, scheduler, num_epochs, device, exp_name):
    """Loop de entrenamiento y validación estándar con mAP extendido por clase."""
    model.to(device)
    history = {"train_loss": [], "val_mAP": [], "val_mAP_per_class": []}
    best_map = -1.0

    for epoch in range(1, num_epochs + 1):
        # --- Fase de Entrenamiento ---
        model.train()
        epoch_losses = []

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_for_type for loss_for_type in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_losses.append(losses.item())

        mean_train_loss = np.mean(epoch_losses)
        scheduler.step()

        # --- Fase de Validación ---
        model.eval()
        metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

        with torch.no_grad():
            for images, targets in valid_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                predictions = model(images)
                metric.update(predictions, targets)

        metrics_results = metric.compute()
        current_map = metrics_results["map_50"].item()
        per_class = metrics_results["map_per_class"].cpu().numpy()

        history["train_loss"].append(mean_train_loss)
        history["val_mAP"].append(current_map)
        history["val_mAP_per_class"].append(per_class)

        print(f"[{exp_name}] Epoch {epoch}/{num_epochs} -> Train Loss: {mean_train_loss:.4f} | Val mAP@0.5: {current_map:.4f}")

        if current_map > best_map:
            best_map = current_map
            checkpoint_path = ROOT / "dev" / f"{exp_name}_best.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

    return history

def visualize_predictions_vs_ground_truth(model, dataset, class_names, device, num_images=8):
    """Muestra una grilla comparativa contrastando inferencias de la red contra etiquetas reales."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    indices = random.sample(range(len(dataset)), num_images)

    for idx, sample_idx in enumerate(indices):
        img_tensor, target = dataset[sample_idx]

        model.eval()
        with torch.no_grad():
            prediction = model([img_tensor.to(device)])[0]

        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img_pil)

        # 1. Dibujar Bounding Boxes Reales (Ground Truth) en amarillo
        for box, lbl in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=2)
            cls_idx = lbl.item() - dataset.label_offset
            draw.text((x1 + 3, y1 + 3), f"GT: {class_names[cls_idx]}", fill="yellow")

        # 2. Dibujar Bounding Boxes Predichos en cian (filtrado por confianza > 0.5)
        CONF_THRESHOLD = 0.5
        pred_boxes = prediction["boxes"].cpu()
        pred_labels = prediction["labels"].cpu()
        pred_scores = prediction["scores"].cpu()

        for box, lbl, score in zip(pred_boxes, pred_labels, pred_scores):
            if score.item() >= CONF_THRESHOLD:
                x1, y1, x2, y2 = box.tolist()
                draw.rectangle([x1, y1, x2, y2], outline="cyan", width=2)
                cls_idx = lbl.item() - dataset.label_offset
                draw.text((x1 + 3, y2 - 12), f"PR: {class_names[cls_idx]} ({score.item():.2f})", fill="cyan")

        axes[idx].imshow(img_pil)
        axes[idx].axis("off")
        axes[idx].set_title(f"Sample Index {sample_idx}", fontsize=9)

    plt.suptitle("Análisis Visual de Errores: Ground Truth (Amarillo) vs Predicciones (Cian)", fontsize=14)
    plt.tight_layout()
    plt.savefig("dev/analisis_errores.png")

# =====================================================================
# BLOQUE PROTECTOR PRINCIPAL DE WINDOWS (Ejecución unificada protegida)
# =====================================================================
if __name__ == '__main__':
    # Inicialización de entorno
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plt.rcParams["figure.figsize"] = (12, 6)

    # Cargar nombres de clases desde data.yaml de forma dinámica
    with open(PROCESSED_ROOT / "data.yaml", "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    CLASS_NAMES = yaml_data.get("names", [])
    NUM_CLASSES = len(CLASS_NAMES)

    print(f"Dispositivo detectado: {device} | Clases cargadas: {CLASS_NAMES} ({NUM_CLASSES})")

    # --- Sección 3: Análisis de Desbalance ---
    train_df = pd.read_csv(SPLIT_CSV["train"])
    class_counts = Counter()

    for _, row in train_df.iterrows():
        annotations = read_yolo_annotations(PROCESSED_ROOT / row["label"])
        for cls_id, *_ in annotations:
            class_counts[cls_id] += 1

    for i in range(NUM_CLASSES):
        if i not in class_counts:
            class_counts[i] = 0

    classes = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    counts = [class_counts[i] for i in range(NUM_CLASSES)]

    plt.figure()
    plt.bar(classes, counts, color="teal")
    plt.title("Distribución de Instancias por Clase en Train Set")
    plt.xlabel("Patrón de Vela")
    plt.ylabel("Cantidad de Cajas")
    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(CURRENT_DIR / "distribucion_instancias.png")

    total_instances = sum(class_counts.values())
    class_weights = {i: (total_instances / (NUM_CLASSES * class_counts[i]) if class_counts[i] > 0 else 1.0) for i in range(NUM_CLASSES)}
    print(f"Pesos de Clase calculados (Frecuencia Inversa): {class_weights}")

    # =====================================================================
    # EJECUCIÓN: COMPARATIVA DE FOCAL LOSS (100 ÉPOCAS)
    # =====================================================================
    NUM_EPOCHS = 100  

    # --- Sección 7: Experimento 2 — Faster R-CNN con Focal Loss ---
    print("\n" + "="*50 + "\nIniciado Experimento 2: Faster R-CNN con Focal Loss (100 Épocas)\n" + "="*50)
    train_dataset_exp2 = CandlestickDetectionDataset(SPLIT_CSV["train"], PROCESSED_ROOT, label_offset=1)
    valid_dataset_exp2 = CandlestickDetectionDataset(SPLIT_CSV["valid"], PROCESSED_ROOT, label_offset=1)

    train_loader_exp2 = DataLoader(train_dataset_exp2, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=detection_collate_fn)
    valid_loader_exp2 = DataLoader(valid_dataset_exp2, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=detection_collate_fn)

    model_exp2 = build_faster_rcnn_model(num_classes=7)
    optimizer_exp2 = torch.optim.SGD(model_exp2.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler_exp2 = torch.optim.lr_scheduler.StepLR(optimizer_exp2, step_size=25, gamma=0.5)

    import torchvision.models.detection.roi_heads as _rh
    _original_loss = _rh.fastrcnn_loss

    def _focal_loss_fn(class_logits, box_regression, labels, regression_targets):
        labels_cat = torch.cat(labels)
        regression_targets_cat = torch.cat(regression_targets)
        classification_loss = softmax_focal_loss(class_logits, labels_cat, gamma=2.0, alpha=0.25)
        sampled_pos = torch.where(labels_cat > 0)[0]
        labels_pos = labels_cat[sampled_pos]
        N = class_logits.shape[0]
        box_regression = box_regression.reshape(N, -1, 4)
        box_loss = F.smooth_l1_loss(box_regression[sampled_pos, labels_pos], regression_targets_cat[sampled_pos], beta=1/9, reduction='sum') / max(labels_cat.numel(), 1)
        return classification_loss, box_loss

    _rh.fastrcnn_loss = _focal_loss_fn
    # Se entrena temporalmente asignando un identificador para su posterior análisis
    history_exp2 = train_detection_model(model_exp2, train_loader_exp2, valid_loader_exp2, optimizer_exp2, scheduler_exp2, NUM_EPOCHS, device, "temp_exp2_focal")
    _rh.fastrcnn_loss = _original_loss  

    # --- Sección 9: Experimento 4 — RetinaNet con Focal Loss Nativa ---
    print("\n" + "="*50 + "\nIniciando Experimento 4: RetinaNet con Focal Loss Nativa (100 Épocas)\n" + "="*50)
    train_dataset_exp4 = CandlestickDetectionDataset(SPLIT_CSV["train"], PROCESSED_ROOT, label_offset=0)
    valid_dataset_exp4 = CandlestickDetectionDataset(SPLIT_CSV["valid"], PROCESSED_ROOT, label_offset=0)

    train_loader_exp4 = DataLoader(train_dataset_exp4, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=detection_collate_fn)
    valid_loader_exp4 = DataLoader(valid_dataset_exp4, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=detection_collate_fn)

    model_exp4 = build_retinanet_model(num_classes=6, focal_gamma=2.0, focal_alpha=0.25)
    optimizer_exp4 = torch.optim.SGD(model_exp4.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0005)
    scheduler_exp4 = torch.optim.lr_scheduler.StepLR(optimizer_exp4, step_size=25, gamma=0.5)

    history_exp4 = train_detection_model(model_exp4, train_loader_exp4, valid_loader_exp4, optimizer_exp4, scheduler_exp4, NUM_EPOCHS, device, "temp_exp4_retinanet_focal")

    # =====================================================================
    # CONSIDERACIÓN 1: GRÁFICOS REALES CON ESCALA Y UNIFICADA (0.0 A 0.8)
    # =====================================================================
    print("\n" + "="*50 + "\nGENERANDO GRÁFICOS UNIFICADOS (ESCALA 0.0 - 0.8)\n" + "="*50)
    epochs_range = range(1, NUM_EPOCHS + 1)
    legends = ["Exp2: FasterRCNN Focal", "Exp4: RetinaNet Focal"]
    histories = [history_exp2, history_exp4]

    # Gráfico 1 — Train Loss
    plt.figure(figsize=(12, 5))
    for h in histories: plt.plot(epochs_range, h["train_loss"], linewidth=2)
    plt.title("Evolución de Train Loss", fontsize=14)
    plt.xlabel("Épocas"); plt.ylabel("Loss"); plt.legend(legends); plt.grid(True, alpha=0.3)
    plt.savefig(CURRENT_DIR / "curvas_aprendizaje_train_loss.png")
    
    # Gráfico 2 — mAP@0.5 General (Unificado a 0.8)
    plt.figure(figsize=(12, 5))
    for h in histories: plt.plot(epochs_range, h["val_mAP"], linewidth=2)
    plt.title("Evolución de mAP@0.5 General", fontsize=14)
    plt.xlabel("Épocas"); plt.ylabel("mAP@0.5"); plt.legend(legends); plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 0.8)  # Forzar escala unificada
    plt.savefig(CURRENT_DIR / "curvas_aprendizaje_val_mAP.png")

    # Gráficos 3 a 8 — mAP por Clase en malla 2x3 (Todos fijos de 0.0 a 0.8)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i in range(NUM_CLASSES):
        ax = axes[i]
        for h in histories:
            class_curve = [epoch_per_class[i] for epoch_per_class in h["val_mAP_per_class"]]
            ax.plot(epochs_range, class_curve, linewidth=1.5)
        ax.set_title(f"mAP@0.5 - {CLASS_NAMES[i]}", fontsize=12)
        ax.set_xlabel("Épocas"); ax.set_ylabel("mAP@0.5")
        ax.set_ylim(0.0, 0.8)  # CONSIDERACIÓN 1: El mismo eje Y estricto para todos
        ax.legend(legends, fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(CURRENT_DIR / "curvas_aprendizaje_val_mAP_per_class.png")

    # =====================================================================
    # CONSIDERACIÓN 2 & 3: SELECCIÓN, TABLA EN MARKDOWN Y EVACUACIÓN DE ARCHIVOS
    # =====================================================================
    map_exp2 = history_exp2["val_mAP"][-1]
    map_exp4 = history_exp4["val_mAP"][-1]
    
    # Evaluar cuál fue matemáticamente el mejor
    if map_exp2 > map_exp4:
        winner_name = "exp2_faster_rcnn_focal"
        winner_temp_flag = "temp_exp2_focal"
        loser_temp_flag = "temp_exp4_retinanet_focal"
        production_model = build_faster_rcnn_model(num_classes=7)
        test_label_offset = 1
        mark_exp2, mark_exp4 = "🏆 **GANADOR (MEJOR mAP)**", ""
    else:
        winner_name = "exp4_retinanet_focal"
        winner_temp_flag = "temp_exp4_retinanet_focal"
        loser_temp_flag = "temp_exp2_focal"
        production_model = build_retinanet_model(num_classes=6, focal_gamma=2.0, focal_alpha=0.25)
        test_label_offset = 0
        mark_exp2, mark_exp4 = "", "🏆 **GANADOR (MEJOR mAP)**"

    # CONSIDERACIÓN 2: Almacenar los resultados en una tabla estructurada Markdown (.md)
    report_path = CURRENT_DIR / "reporte_experimentos.md"
    markdown_content = f"""# Reporte de Rendimiento - Comparativa Focal Loss (100 Épocas)

| Experimento | Modelo Base | Estrategia de Balanceo | Final Train Loss | Validación mAP@0.5 | Estado |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Exp 2** | Faster R-CNN | Focal Loss (Patch) | {history_exp2['train_loss'][-1]:.4f} | {map_exp2:.4f} | {mark_exp2} |
| **Exp 4** | RetinaNet | Focal Loss (Nativa) | {history_exp4['train_loss'][-1]:.4f} | {map_exp4:.4f} | {mark_exp4} |

*Nota: Reporte automatizado generado de forma nativa por el pipeline de entrenamiento el 21/06/2026.*
"""
    report_path.write_text(markdown_content, encoding="utf-8")
    print(f"\nTabla de rendimiento guardada exitosamente en: {report_path}")

    # CONSIDERACIÓN 3: Consolidar solo el ganador con el nombre del experimento y purgar los peores
    winner_temp_file = CURRENT_DIR / f"{winner_temp_flag}_best.pth"
    loser_temp_file = CURRENT_DIR / f"{loser_temp_flag}_best.pth"
    
    final_winner_path = CURRENT_DIR / f"{winner_name}_best.pth"
    
    # Renombrar el mejor al nombre de su experimento
    if winner_temp_file.exists():
        winner_temp_file.rename(final_winner_path)
    
    # Eliminar el peor para no dejar basura colgada ni saturar el disco
    if loser_temp_file.exists():
        loser_temp_file.unlink()
        print(f"Purgado de disco: Se eliminó el checkpoint del experimento perdedor ({loser_temp_flag}_best.pth).")

    print(f"🏆 Modelo consolidado único guardado exitosamente como: {final_winner_path.name}")

    # --- Evaluación Final en Test Set usando el archivo guardado legítimo ---
    production_model.load_state_dict(torch.load(final_winner_path, map_location=device))
    production_model.to(device)
    production_model.eval()

    test_dataset = CandlestickDetectionDataset(SPLIT_CSV["test"], PROCESSED_ROOT, label_offset=test_label_offset)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=detection_collate_fn)

    test_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            test_metric.update(production_model(images), targets)

    test_results = test_metric.compute()
    print("\n" + "="*50 + "\n=== EVALUACIÓN DEFINITIVA EN TEST SET ===\n" + "="*50)
    print(f"mAP@0.5 General obtenido en Test: {test_results['map_50'].item():.4f}")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"mAP@0.5 Desglosado -> Clase [{class_name}]: {test_results['map_per_class'][i].item():.4f}")

    # --- Diagnóstico Visual de Errores y Exportación a producción ---
    visualize_predictions_vs_ground_truth(production_model, test_dataset, CLASS_NAMES, device, num_images=8)
    
    # Duplicar el ganador como 'modelo.pth' para que Streamlit mantenga su compatibilidad directa
    torch.save(production_model.state_dict(), CURRENT_DIR / "modelo.pth")
    print("¡Proceso finalizado con éxito! El Ryzen 7 ya puede proceder al apagado seguro.")