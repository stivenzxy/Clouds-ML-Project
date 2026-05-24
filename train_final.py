"""
Punto C — Entrenamiento del modelo final.

Entrena DenseNet-121 (arquitectura ganadora del punto B) sobre el conjunto
de entrenamiento completo (`train_split.csv`, 70%) y evalúa al final de cada
epoch sobre el conjunto de validación (`val_split.csv`, 15%, intocado hasta
ahora).

Configuración:
- 25 epochs con early stopping (patience=5) sobre F1 macro de validación.
- CosineAnnealingLR: reduce LR de 1e-4 a ~1e-6 siguiendo curva coseno.
- BCEWithLogitsLoss + Adam.
- Mixed Precision (AMP).
- Guarda el mejor checkpoint según F1 macro en validación.
- Logueo a Weights & Biases.

Salida:
- `best_model_final_DenseNet121.pth`: pesos del mejor modelo según val F1.
- Logs en W&B (offline en `wandb/offline-run-*`).
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('WANDB_MODE', 'offline')

import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
from tqdm import tqdm
import wandb

from dataset import CloudDataset, get_train_transforms, get_valid_transforms
from model import get_densenet121


SEED = 42
EPOCHS = 25
PATIENCE = 5            # Early stopping: epochs sin mejora antes de cortar.
LEARNING_RATE = 1e-4
MIN_LR = 1e-6
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL = 64
NUM_WORKERS = 6
IMG_DIR = 'data/train_images'

BEST_MODEL_PATH = 'best_model_final_DenseNet121.pth'


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds)

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    avg_loss = running_loss / len(loader.dataset)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return avg_loss, f1_macro, f1_weighted


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # ---- Datos ----
    train_df = pd.read_csv('data/train_split.csv')
    val_df = pd.read_csv('data/val_split.csv')

    print(f"Train: {len(train_df)} imágenes")
    print(f"Val:   {len(val_df)} imágenes")

    train_ds = CloudDataset(train_df, IMG_DIR, transforms=get_train_transforms())
    val_ds = CloudDataset(val_df, IMG_DIR, transforms=get_valid_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE_VAL, shuffle=False,
        num_workers=max(2, NUM_WORKERS // 2), pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    # ---- Modelo, loss, optimizador, scheduler, scaler ----
    model = get_densenet121(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)
    scaler = GradScaler('cuda')

    # ---- W&B ----
    wandb.init(
        project="proyecto-nubes",
        name="final-DenseNet121",
        config={
            "architecture": "DenseNet121",
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "lr_init": LEARNING_RATE,
            "lr_min": MIN_LR,
            "scheduler": "CosineAnnealingLR",
            "batch_size_train": BATCH_SIZE_TRAIN,
            "batch_size_val": BATCH_SIZE_VAL,
            "img_size": "320x480",
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss",
            "amp": True,
            "seed": SEED,
        },
    )

    # ---- Loop con early stopping ----
    best_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    print(f"\n{'=' * 60}")
    print(f"Entrenamiento del modelo final (DenseNet121)")
    print(f"{'=' * 60}")

    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        epoch_start = time.time()

        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        t_train = time.time() - t0

        t0 = time.time()
        val_loss, f1_m, f1_w = validate(model, val_loader, criterion, device)
        t_val = time.time() - t0

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "f1_macro": f1_m,
            "f1_weighted": f1_w,
            "lr": current_lr,
            "epoch_time_sec": epoch_time,
        })

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "f1_macro": f1_m,
            "f1_weighted": f1_w,
            "lr": current_lr,
        })

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  F1 Macro:  {f1_m:.4f}  | F1 Weighted: {f1_w:.4f}")
        print(f"  LR actual: {current_lr:.2e} | Tiempo: train {t_train:.1f}s | val {t_val:.1f}s")

        # Checkpoint sobre F1 macro de validación.
        if f1_m > best_f1:
            best_f1 = f1_m
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  ✓ Nuevo mejor F1 macro: {f1_m:.4f} → checkpoint guardado")
        else:
            epochs_no_improve += 1
            print(f"  Sin mejora ({epochs_no_improve}/{PATIENCE} epochs)")

        # Early stopping.
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping en epoch {epoch}. Mejor F1: {best_f1:.4f} (epoch {best_epoch})")
            break

        scheduler.step()

    total_time = time.time() - train_start

    # ---- Resumen ----
    print(f"\n{'=' * 60}")
    print(f"Entrenamiento finalizado en {total_time / 60:.1f} min")
    print(f"Mejor F1 Macro (val): {best_f1:.4f} en epoch {best_epoch}")
    print(f"Checkpoint guardado en: {BEST_MODEL_PATH}")
    print(f"{'=' * 60}")

    # Guardar historial para el paper.
    history_df = pd.DataFrame(history)
    history_df.to_csv('training_history_final.csv', index=False)
    print("Historial de entrenamiento guardado en: training_history_final.csv")

    # Reportar a W&B.
    wandb.summary['best_f1_macro'] = best_f1
    wandb.summary['best_epoch'] = best_epoch
    wandb.summary['total_time_min'] = total_time / 60
    wandb.finish()


if __name__ == "__main__":
    main()
