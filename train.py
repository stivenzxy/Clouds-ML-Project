import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import os

from dataset import CloudDataset, get_train_transforms, get_valid_transforms
from model import get_mobilenetv3_large

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
    
    # Cargar DataFrames
    train_df = pd.read_csv('data/train_split.csv')
    val_df = pd.read_csv('data/val_split.csv')
    img_dir = 'data/train_images'
    
    # Datasets y Dataloaders
    train_ds = CloudDataset(train_df, img_dir, transforms=get_train_transforms())
    val_ds = CloudDataset(val_df, img_dir, transforms=get_valid_transforms())
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Modelo, Pérdida y Optimizado
    model = get_mobilenetv3_large(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Bucle de entrenamiento
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, f1_m, f1_w = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"F1 Macro: {f1_m:.4f} | F1 Weighted: {f1_w:.4f}")
        
    # Guardar modelo
    torch.save(model.state_dict(), "cloud_model.pth")
    print("\nEntrenamiento finalizado y modelo guardado.")

if __name__ == "__main__":
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    main()
