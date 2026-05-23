import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CloudDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms
        self.img_ids = self.df['image_id'].values
        self.labels = self.df[['Fish', 'Flower', 'Gravel', 'Sugar']].values

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id)
        
        # Leer imagen
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Etiquetas (multi-etiqueta)
        label = self.labels[idx].astype('float32')
        
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        
        return image, torch.tensor(label)

def get_train_transforms():
    """
    Transformaciones para el set de entrenamiento.
    Incluye: Aumentos de orientación, brillo/contraste y normalización.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Resize(320, 480), # Redimensionar para acelerar el entrenamiento (opcional pero recomendado)
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_valid_transforms():
    """
    Transformaciones para validación/test. Solo normalización y redimensionamiento.
    """
    return A.Compose([
        A.Resize(320, 480),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

if __name__ == "__main__":
    # Prueba rápida
    train_df = pd.read_csv('data/train_split.csv')
    train_dir = 'data/train_images'
    
    dataset = CloudDataset(train_df, train_dir, transforms=get_train_transforms())
    img, label = dataset[0]
    print(f"Imagen shape: {img.shape}")
    print(f"Etiquetas: {label}")
