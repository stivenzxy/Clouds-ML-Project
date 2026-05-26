import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import CloudDataset, get_valid_transforms
from model import get_densenet121


CLASSES = ['Fish', 'Flower', 'Gravel', 'Sugar']
BEST_MODEL_PATH = 'models/best_model_final_DenseNet121.pth'
IMG_DIR = 'data/train_images'
RESULTS_DIR = Path('results')
THRESHOLD = 0.5


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Inferencia"):
        images = images.to(device, non_blocking=True)
        with autocast(device_type='cuda'):
            outputs = model(images)
        probs = torch.sigmoid(outputs).float().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs > THRESHOLD).astype(int)
    return probs, preds, labels


def compute_metrics_per_class(y_true, y_pred):
    rows = []
    cms = {}

    for i, cls in enumerate(CLASSES):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        acc = accuracy_score(y_t, y_p)
        prec = precision_score(y_t, y_p, zero_division=0)
        rec = recall_score(y_t, y_p, zero_division=0)
        f1 = f1_score(y_t, y_p, zero_division=0)
        # cm = [[TN, FP], [FN, TP]]
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])

        rows.append({
            'Clase': cls,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'TN': int(cm[0, 0]),
            'FP': int(cm[0, 1]),
            'FN': int(cm[1, 0]),
            'TP': int(cm[1, 1]),
            'Soporte (positivos)': int(y_t.sum()),
            'Soporte (negativos)': int((y_t == 0).sum()),
        })
        cms[cls] = cm

    df = pd.DataFrame(rows)

    # Promedio macro (no ponderado por soporte).
    macro_row = {
        'Clase': 'MACRO',
        'Accuracy': df['Accuracy'].mean(),
        'Precision': df['Precision'].mean(),
        'Recall': df['Recall'].mean(),
        'F1': df['F1'].mean(),
        'TN': '-', 'FP': '-', 'FN': '-', 'TP': '-',
        'Soporte (positivos)': int(y_true.sum()),
        'Soporte (negativos)': int((y_true == 0).sum()),
    }
    df = pd.concat([df, pd.DataFrame([macro_row])], ignore_index=True)
    return df, cms


def plot_confusion_matrices(cms, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, cls in zip(axes, CLASSES):
        cm = cms[cls]
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title(f'{cls}', fontsize=13, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicho: NO', 'Predicho: SÍ'])
        ax.set_yticklabels(['Real: NO', 'Real: SÍ'])
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real')

        threshold = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > threshold else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color=color, fontsize=14, fontweight='bold')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Matrices de confusión por clase — Modelo final (DenseNet121) sobre val_split',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura guardada en: {save_path}")


def export_markdown_table(df, save_path):
    rows = []
    rows.append("| Clase | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN |")
    rows.append("|-------|----------|-----------|--------|-----|----|----|----|----|")
    for _, r in df.iterrows():
        if r['Clase'] == 'MACRO':
            rows.append(f"| **{r['Clase']}** | **{r['Accuracy']:.4f}** | **{r['Precision']:.4f}** | **{r['Recall']:.4f}** | **{r['F1']:.4f}** | - | - | - | - |")
        else:
            rows.append(
                f"| {r['Clase']} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | "
                f"{r['Recall']:.4f} | {r['F1']:.4f} | {r['TP']} | {r['FP']} | {r['FN']} | {r['TN']} |"
            )
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("# Resultados del modelo final — DenseNet121 sobre val_split\n\n")
        f.write("\n".join(rows))
        f.write("\n")
    print(f"Tabla markdown guardada en: {save_path}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    if not Path(BEST_MODEL_PATH).exists():
        raise FileNotFoundError(
            f"No se encontró {BEST_MODEL_PATH}. Ejecutá train_final.py primero."
        )

    val_df = pd.read_csv('data/val_split.csv')
    print(f"Val:   {len(val_df)} imágenes")

    val_ds = CloudDataset(val_df, IMG_DIR, transforms=get_valid_transforms())
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    print(f"Cargando checkpoint: {BEST_MODEL_PATH}")
    model = get_densenet121(pretrained=False).to(device)
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    probs, preds, labels = predict(model, val_loader, device)

    metrics_df, cms = compute_metrics_per_class(labels, preds)

    print("\n" + "=" * 80)
    print("MÉTRICAS POR CLASE — Modelo final (DenseNet121) sobre val_split")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    print("=" * 80)

    metrics_df.to_csv(RESULTS_DIR / 'metrics_per_class.csv', index=False)
    print(f"\nMétricas guardadas en: {RESULTS_DIR / 'metrics_per_class.csv'}")

    plot_confusion_matrices(cms, RESULTS_DIR / 'confusion_matrices.png')
    export_markdown_table(metrics_df, RESULTS_DIR / 'metrics_summary.md')

    # Predicciones completas (probs + binarias) por imagen, para inspección.
    pred_df = val_df[['image_id']].copy()
    for i, cls in enumerate(CLASSES):
        pred_df[f'{cls}_true'] = labels[:, i].astype(int)
        pred_df[f'{cls}_prob'] = probs[:, i]
        pred_df[f'{cls}_pred'] = preds[:, i]
    pred_df.to_csv(RESULTS_DIR / 'predictions_val.csv', index=False)
    print(f"Predicciones guardadas en: {RESULTS_DIR / 'predictions_val.csv'}")


if __name__ == "__main__":
    main()
