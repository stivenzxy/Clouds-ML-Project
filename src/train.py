import os
# Conflicto OpenMP entre librerías en algunos entornos Windows.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# W&B offline: después se sincroniza con `wandb sync`.
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
from sklearn.metrics import f1_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tqdm import tqdm
import wandb

from dataset import CloudDataset, get_train_transforms, get_valid_transforms
from model import get_mobilenetv3_large, get_densenet121


SEED = 42


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

        # AMP: forward en float16 donde sea seguro.
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

        # Umbral 0.5 sobre la sigmoide para obtener predicciones binarias.
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

    batch_size_train = 32   # Bajar a 24/16 si hay CUDA OOM.
    batch_size_val = 64
    num_workers = 6
    epochs = 5
    learning_rate = 1e-4
    img_dir = 'data/train_images'

    # CV se hace SOLO sobre train_split (70%). Val/test no se tocan acá.
    full_train_df = pd.read_csv('data/train_split.csv')
    labels_arr = full_train_df[['Fish', 'Flower', 'Gravel', 'Sugar']].values

    architectures = {
        'MobileNetV3': get_mobilenetv3_large,
        'DenseNet121': get_densenet121,
    }

    mskf = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)

    results = []

    for arch_name, get_model_fn in architectures.items():
        print(f"\n{'=' * 40}")
        print(f"Arquitectura: {arch_name}")
        print(f"{'=' * 40}")

        arch_f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(mskf.split(full_train_df, labels_arr)):
            print(f"\n--- Fold {fold + 1}/2 ---")
            fold_start = time.time()

            wandb.init(
                project="proyecto-nubes",
                name=f"{arch_name}-fold{fold + 1}",
                group=arch_name,
                job_type="train",
                config={
                    "architecture": arch_name,
                    "fold": fold + 1,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "batch_size_train": batch_size_train,
                    "batch_size_val": batch_size_val,
                    "img_size": "320x480",
                    "optimizer": "Adam",
                    "loss": "BCEWithLogitsLoss",
                    "amp": True,
                    "seed": SEED,
                },
                reinit=True,
            )

            train_df = full_train_df.iloc[train_idx].reset_index(drop=True)
            val_df = full_train_df.iloc[val_idx].reset_index(drop=True)

            train_ds = CloudDataset(train_df, img_dir, transforms=get_train_transforms())
            val_ds = CloudDataset(val_df, img_dir, transforms=get_valid_transforms())

            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size_train,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size_val,
                shuffle=False,
                num_workers=max(2, num_workers // 2),
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
            )

            model = get_model_fn(pretrained=True).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scaler = GradScaler('cuda')

            best_f1_macro = 0.0

            for epoch in range(epochs):
                print(f"\nEpoch {epoch + 1}/{epochs}")
                epoch_start = time.time()

                t_train_start = time.time()
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
                t_train = time.time() - t_train_start

                t_val_start = time.time()
                val_loss, f1_m, f1_w = validate(model, val_loader, criterion, device)
                t_val = time.time() - t_val_start

                epoch_time = time.time() - epoch_start
                current_lr = optimizer.param_groups[0]['lr']

                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "f1_macro": f1_m,
                    "f1_weighted": f1_w,
                    "lr": current_lr,
                    "epoch_time_sec": epoch_time,
                    "train_time_sec": t_train,
                    "val_time_sec": t_val,
                })

                print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"  F1 Macro:  {f1_m:.4f}  | F1 Weighted: {f1_w:.4f}")
                print(f"  Tiempo:    train {t_train:.1f}s | val {t_val:.1f}s | total {epoch_time:.1f}s")

                if f1_m > best_f1_macro:
                    best_f1_macro = f1_m
                    model_path = f"models/best_model_{arch_name}_fold{fold + 1}.pth"
                    torch.save(model.state_dict(), model_path)
                    print(f"  ✓ Nuevo mejor F1 ({f1_m:.4f}) - checkpoint guardado: {model_path}")

            arch_f1_scores.append(best_f1_macro)
            fold_time = time.time() - fold_start
            print(f"\n  Fold {fold + 1} completado en {fold_time / 60:.1f} min — Mejor F1: {best_f1_macro:.4f}")
            wandb.finish()

            del model, optimizer, scaler, train_loader, val_loader, train_ds, val_ds
            torch.cuda.empty_cache()

        mean_f1 = float(np.mean(arch_f1_scores))
        std_f1 = float(np.std(arch_f1_scores))
        results.append({
            "Architecture": arch_name,
            "F1 Macro (Mean)": mean_f1,
            "F1 Macro (Std)": std_f1,
            "Folds": arch_f1_scores,
        })

        print(f"\n>>> {arch_name}: F1 Macro = {mean_f1:.4f} ± {std_f1:.4f}")

    print("\n" + "=" * 60)
    print("RESULTADOS FINALES — Validación cruzada estratificada (2 folds)")
    print("=" * 60)
    print(f"{'Arquitectura':<20} {'F1 Macro (Media ± Std)':<30}")
    print("-" * 60)
    for res in results:
        print(f"{res['Architecture']:<20} {res['F1 Macro (Mean)']:.4f} ± {res['F1 Macro (Std)']:.4f}")
    print("=" * 60)

    wandb.init(project="proyecto-nubes", name="cv-summary", job_type="aggregate", reinit=True)
    table = wandb.Table(columns=["Architecture", "F1 Macro (Mean ± Std)", "F1 Macro Mean", "F1 Macro Std", "Fold 1", "Fold 2"])
    for res in results:
        mean_std_str = f"{res['F1 Macro (Mean)']:.4f} ± {res['F1 Macro (Std)']:.4f}"
        table.add_data(
            res["Architecture"],
            mean_std_str,
            res["F1 Macro (Mean)"],
            res["F1 Macro (Std)"],
            res["Folds"][0],
            res["Folds"][1],
        )
    wandb.log({"CV Summary Table": table})
    wandb.finish()

    print("\nValidación cruzada finalizada.")


if __name__ == "__main__":
    main()
