import os
os.environ['WANDB_MODE'] = 'offline'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tqdm import tqdm
import wandb

from dataset import CloudDataset, get_train_transforms, get_valid_transforms
from model import get_mobilenetv3_large, get_densenet121

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            # Aplicar sigmoid y threshold para métricas
            preds = torch.sigmoid(outputs).cpu().numpy()
            preds = (preds > 0.5).astype(int)
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds)
            
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    
    avg_loss = running_loss / len(loader.dataset)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, f1_macro, f1_weighted

def main():
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    epochs = 5
    learning_rate = 1e-4
    img_dir = 'data/train_images'
    
    # Cargar DataFrames
    full_df = pd.read_csv('data/train_multilabel.csv')
    
    # Arquitecturas a entrenar
    architectures = {
        'MobileNet3': get_mobilenetv3_large,
        'densenet121': get_densenet121
    }
    
    # Configuración de K-Fold Estratificado Multietiqueta
    mskf = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    
    # Etiquetas para la estratificación
    labels = full_df[['Fish', 'Flower', 'Gravel', 'Sugar']].values
    
    # Para almacenar resultados finales
    results = []
    
    for arch_name, get_model_fn in architectures.items():
        print(f"\n{'='*30}")
        print(f"Arquitectura: {arch_name}")
        print(f"{'='*30}")
        
        arch_f1_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(mskf.split(full_df, labels)):
            print(f"\n--- Fold {fold+1}/2 ---")
            
            # Inicializar W&B para cada fold
            wandb.init(
                project="proyecto-nubes",
                name=f"{arch_name}-fold{fold+1}",
                config={
                    "architecture": arch_name,
                    "fold": fold + 1,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "batch_size": batch_size,
                },
                reinit=True
            )
            
            train_df = full_df.iloc[train_idx].reset_index(drop=True)
            val_df = full_df.iloc[val_idx].reset_index(drop=True)
            
            # Datasets y Dataloaders
            train_ds = CloudDataset(train_df, img_dir, transforms=get_train_transforms())
            val_ds = CloudDataset(val_df, img_dir, transforms=get_valid_transforms())
            
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
            
            # Modelo, Pérdida y Optimizado
            model = get_model_fn(pretrained=True).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            best_val_loss = float('inf')
            best_f1_macro = 0.0
            
            # Bucle de entrenamiento
            for epoch in range(epochs):
                print(f"Epoch {epoch+1}/{epochs}")
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, f1_m, f1_w = validate(model, val_loader, criterion, device)
                
                # Obtener LR actual
                current_lr = optimizer.param_groups[0]['lr']
                
                # Loguear a W&B
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "f1_macro": f1_m,
                    "f1_weighted": f1_w,
                    "lr": current_lr
                })
                
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"F1 Macro: {f1_m:.4f} | F1 Weighted: {f1_w:.4f}")
                
                if f1_m > best_f1_macro:
                    best_f1_macro = f1_m
                    model_path = f"best_model_{arch_name}_fold{fold+1}.pth"
                    torch.save(model.state_dict(), model_path)
                    print(f"Mejor modelo (F1: {f1_m:.4f}) guardado en {model_path}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            
            arch_f1_scores.append(best_f1_macro)
            wandb.finish()
            
        # Calcular media y std por arquitectura
        mean_f1 = np.mean(arch_f1_scores)
        std_f1 = np.std(arch_f1_scores)
        results.append({
            "Architecture": arch_name,
            "F1 Macro (Mean)": mean_f1,
            "F1 Macro (Std)": std_f1
        })
    
    # Reportar tabla final a W&B (iniciar un run final para el reporte)
    wandb.init(project="proyecto-nubes", name="final-results")
    table = wandb.Table(columns=["Architecture", "F1 Macro Mean", "F1 Macro Std"])
    for res in results:
        table.add_data(res["Architecture"], res["F1 Macro (Mean)"], res["F1 Macro (Std)"])
    
    wandb.log({"Final F1 Summary": table})
    wandb.finish()
    
    print("\nEntrenamiento de validación cruzada finalizado.")

if __name__ == "__main__":
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '1'
    main()
