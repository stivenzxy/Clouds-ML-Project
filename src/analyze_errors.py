import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from dataset import CloudDataset, get_valid_transforms
from model import get_densenet121


CLASSES = ['Fish', 'Flower', 'Gravel', 'Sugar']
BEST_MODEL_PATH = 'models/best_model_final_DenseNet121.pth'
IMG_DIR = Path('data/train_images')
RESULTS_DIR = Path('results')
THRESHOLD = 0.5


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in tqdm(loader, desc="Inferencia (test)"):
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
    rows, cms = [], {}
    for i, cls in enumerate(CLASSES):
        y_t, y_p = y_true[:, i], y_pred[:, i]
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        rows.append({
            'Clase': cls,
            'Accuracy': accuracy_score(y_t, y_p),
            'Precision': precision_score(y_t, y_p, zero_division=0),
            'Recall': recall_score(y_t, y_p, zero_division=0),
            'F1': f1_score(y_t, y_p, zero_division=0),
            'TN': int(cm[0, 0]), 'FP': int(cm[0, 1]),
            'FN': int(cm[1, 0]), 'TP': int(cm[1, 1]),
            'Soporte (positivos)': int(y_t.sum()),
            'Soporte (negativos)': int((y_t == 0).sum()),
        })
        cms[cls] = cm

    df = pd.DataFrame(rows)
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


def plot_confusion_matrices(cms, save_path, title_suffix='test_split'):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, cls in zip(axes, CLASSES):
        cm = cms[cls]
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title(f'{cls}', fontsize=13, fontweight='bold')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicho: NO', 'Predicho: SÍ'])
        ax.set_yticklabels(['Real: NO', 'Real: SÍ'])
        ax.set_xlabel('Predicción'); ax.set_ylabel('Real')
        threshold = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > threshold else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color=color, fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'Matrices de confusión por clase — DenseNet121 sobre {title_suffix}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura guardada en: {save_path}")


def export_markdown_table(df, save_path, title='Resultados sobre test_split'):
    rows = [
        "| Clase | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN |",
        "|-------|----------|-----------|--------|-----|----|----|----|----|",
    ]
    for _, r in df.iterrows():
        if r['Clase'] == 'MACRO':
            rows.append(
                f"| **{r['Clase']}** | **{r['Accuracy']:.4f}** | **{r['Precision']:.4f}** | "
                f"**{r['Recall']:.4f}** | **{r['F1']:.4f}** | - | - | - | - |"
            )
        else:
            rows.append(
                f"| {r['Clase']} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | "
                f"{r['Recall']:.4f} | {r['F1']:.4f} | {r['TP']} | {r['FP']} | {r['FN']} | {r['TN']} |"
            )
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"# {title}\n\n")
        f.write("\n".join(rows))
        f.write("\n")
    print(f"Tabla markdown guardada en: {save_path}")


# ---------- Selección de aciertos / errores ----------
def confianza_acierto(probs_row, labels_row):
    # Para positivas queremos prob alta; para negativas, prob baja. Promedio en [0,1].
    aligned = np.where(labels_row == 1, probs_row, 1 - probs_row)
    return float(aligned.mean())


def confianza_error(probs_row, labels_row, preds_row):
    # Suma de distancias al umbral en la dirección equivocada (mayor = error más "seguro").
    mask_wrong = preds_row != labels_row
    if not mask_wrong.any():
        return 0.0
    distances = np.where(preds_row == 1, probs_row - 0.5, 0.5 - probs_row)
    return float(distances[mask_wrong].sum())


def seleccionar_ejemplos(image_ids, probs, preds, labels, n_aciertos=5, n_errores=5):
    n = len(image_ids)
    correct_mask = (preds == labels).all(axis=1)

    # Aciertos: alta confianza, priorizando diversidad de clases positivas.
    correct_idx = np.where(correct_mask)[0]
    conf_aciertos = np.array([confianza_acierto(probs[i], labels[i]) for i in correct_idx])
    order = np.argsort(-conf_aciertos)
    aciertos_sel = []
    classes_cubiertas = set()
    for j in order:
        idx = correct_idx[j]
        positivas = set(np.where(labels[idx] == 1)[0])
        if positivas - classes_cubiertas:
            aciertos_sel.append(idx)
            classes_cubiertas.update(positivas)
        if len(aciertos_sel) >= n_aciertos:
            break
    # Rellenar con los más confiados si quedaron huecos.
    if len(aciertos_sel) < n_aciertos:
        for j in order:
            idx = correct_idx[j]
            if idx not in aciertos_sel:
                aciertos_sel.append(idx)
            if len(aciertos_sel) >= n_aciertos:
                break

    # Errores: los de mayor confianza incorrecta.
    error_idx = np.where(~correct_mask)[0]
    conf_errores = np.array([confianza_error(probs[i], labels[i], preds[i]) for i in error_idx])
    order_err = np.argsort(-conf_errores)
    errores_sel = [error_idx[j] for j in order_err[:n_errores]]

    return aciertos_sel, errores_sel


def render_imagen(ax, image_id, probs_row, labels_row, preds_row, es_acierto):
    img = cv2.imread(str(IMG_DIR / image_id))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)

    lineas = []
    for i, cls in enumerate(CLASSES):
        real = int(labels_row[i])
        pred = int(preds_row[i])
        prob = probs_row[i]
        marca = '✓' if real == pred else '✗'
        lineas.append(f"{cls[:3]}: R={real} P={pred} ({prob:.2f}) {marca}")
    texto = "\n".join(lineas)
    color_borde = 'green' if es_acierto else 'red'
    titulo = ('ACIERTO' if es_acierto else 'ERROR') + f"  —  {image_id}"
    ax.set_title(titulo, fontsize=10, fontweight='bold', color=color_borde)
    ax.text(
        0.02, 0.02, texto, transform=ax.transAxes,
        fontsize=8, family='monospace', color='white',
        verticalalignment='bottom',
        bbox=dict(facecolor='black', alpha=0.65, pad=4, edgecolor='none'),
    )
    for spine in ax.spines.values():
        spine.set_edgecolor(color_borde)
        spine.set_linewidth(2.5)
    ax.set_xticks([]); ax.set_yticks([])


def plot_aciertos_errores(image_ids, probs, preds, labels, aciertos_idx, errores_idx, save_path):
    # Grid 2x5: fila superior aciertos, fila inferior errores.
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))

    for col, idx in enumerate(aciertos_idx):
        render_imagen(axes[0, col], image_ids[idx], probs[idx], labels[idx], preds[idx], es_acierto=True)
    for col, idx in enumerate(errores_idx):
        render_imagen(axes[1, col], image_ids[idx], probs[idx], labels[idx], preds[idx], es_acierto=False)

    fig.suptitle(
        '5 aciertos (arriba) y 5 errores (abajo) sobre test_split — DenseNet121\n'
        'R = etiqueta real, P = predicción, (prob) = probabilidad sigmoide',
        fontsize=13, fontweight='bold', y=1.00,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Figura guardada en: {save_path}")


def exportar_analisis_md(image_ids, probs, preds, labels, aciertos_idx, errores_idx, save_path):
    lineas = ["# Análisis cualitativo — aciertos y errores sobre test_split\n"]

    lineas.append("## Aciertos seleccionados\n")
    lineas.append("| # | image_id | Reales | Predichas | Probabilidades |")
    lineas.append("|---|----------|--------|-----------|----------------|")
    for k, idx in enumerate(aciertos_idx, start=1):
        reales = ", ".join(c for i, c in enumerate(CLASSES) if labels[idx, i] == 1) or "(ninguna)"
        preds_s = ", ".join(c for i, c in enumerate(CLASSES) if preds[idx, i] == 1) or "(ninguna)"
        probs_s = " | ".join(f"{c}={probs[idx, i]:.2f}" for i, c in enumerate(CLASSES))
        lineas.append(f"| {k} | {image_ids[idx]} | {reales} | {preds_s} | {probs_s} |")

    lineas.append("\n## Errores seleccionados\n")
    lineas.append("| # | image_id | Reales | Predichas | Probabilidades | Tipo de error |")
    lineas.append("|---|----------|--------|-----------|----------------|---------------|")
    for k, idx in enumerate(errores_idx, start=1):
        reales = ", ".join(c for i, c in enumerate(CLASSES) if labels[idx, i] == 1) or "(ninguna)"
        preds_s = ", ".join(c for i, c in enumerate(CLASSES) if preds[idx, i] == 1) or "(ninguna)"
        probs_s = " | ".join(f"{c}={probs[idx, i]:.2f}" for i, c in enumerate(CLASSES))
        errores_cls = []
        for i, c in enumerate(CLASSES):
            if preds[idx, i] == 1 and labels[idx, i] == 0:
                errores_cls.append(f"FP {c}")
            elif preds[idx, i] == 0 and labels[idx, i] == 1:
                errores_cls.append(f"FN {c}")
        tipo = "; ".join(errores_cls)
        lineas.append(f"| {k} | {image_ids[idx]} | {reales} | {preds_s} | {probs_s} | {tipo} |")

    lineas.append(
        "\n\n## Cómo usar esta tabla para el paper\n\n"
        "Para cada imagen de error, abrí la figura `aciertos_errores.png` y mirala. "
        "Después redactá 1-2 frases que expliquen por qué creés que el modelo se equivocó, "
        "relacionando lo que ves visualmente con las características esperadas del patrón "
        "(ej. *Fish* tiene filamentos alargados, *Sugar* es disperso y homogéneo, *Flower* "
        "son células redondeadas, *Gravel* son agregaciones granulares).\n"
    )

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lineas))
    print(f"Análisis guardado en: {save_path}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    if not Path(BEST_MODEL_PATH).exists():
        raise FileNotFoundError(
            f"No se encontró {BEST_MODEL_PATH}. Ejecutá train_final.py primero."
        )

    test_df = pd.read_csv('data/test_split.csv')
    print(f"Test:  {len(test_df)} imágenes")

    test_ds = CloudDataset(test_df, str(IMG_DIR), transforms=get_valid_transforms())
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    print(f"Cargando checkpoint: {BEST_MODEL_PATH}")
    model = get_densenet121(pretrained=False).to(device)
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    probs, preds, labels = predict(model, test_loader, device)
    image_ids = test_df['image_id'].values

    metrics_df, cms = compute_metrics_per_class(labels, preds)
    print("\n" + "=" * 80)
    print("MÉTRICAS FINALES SOBRE TEST_SPLIT — DenseNet121")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    print("=" * 80)

    metrics_df.to_csv(RESULTS_DIR / 'metrics_test_per_class.csv', index=False)
    print(f"\nMétricas test guardadas en: {RESULTS_DIR / 'metrics_test_per_class.csv'}")
    export_markdown_table(
        metrics_df, RESULTS_DIR / 'metrics_test_summary.md',
        title='Resultados finales — DenseNet121 sobre test_split (partición imparcial)',
    )
    plot_confusion_matrices(cms, RESULTS_DIR / 'confusion_matrices_test.png')

    pred_df = test_df[['image_id']].copy()
    for i, cls in enumerate(CLASSES):
        pred_df[f'{cls}_true'] = labels[:, i].astype(int)
        pred_df[f'{cls}_prob'] = probs[:, i]
        pred_df[f'{cls}_pred'] = preds[:, i]
    pred_df.to_csv(RESULTS_DIR / 'predictions_test.csv', index=False)
    print(f"Predicciones guardadas en: {RESULTS_DIR / 'predictions_test.csv'}")

    aciertos_idx, errores_idx = seleccionar_ejemplos(image_ids, probs, preds, labels)
    plot_aciertos_errores(
        image_ids, probs, preds, labels, aciertos_idx, errores_idx,
        RESULTS_DIR / 'aciertos_errores.png',
    )
    exportar_analisis_md(
        image_ids, probs, preds, labels, aciertos_idx, errores_idx,
        RESULTS_DIR / 'analisis_aciertos_errores.md',
    )

    n_aciertos_total = int((preds == labels).all(axis=1).sum())
    n_total = len(image_ids)
    print(f"\nAciertos totales (las 4 etiquetas correctas): {n_aciertos_total}/{n_total} "
          f"({100 * n_aciertos_total / n_total:.1f}%)")


if __name__ == '__main__':
    main()
